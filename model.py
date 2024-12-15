import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, APPNP
from torch_geometric.utils import dropout_adj
from copy import deepcopy
import numpy as np

import os
import copy

class GNNBlock(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, output_dim, num_layers):
        super(GNNBlock, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        
        current_in_dim, current_out_dim = input_dim, hidden_dim
        for layer_index in range(num_layers):
            if layer_index == num_layers - 1:
                current_out_dim = output_dim
            # Graph Convolutional Layer
            self.conv_layers.append(GCNConv(current_in_dim, current_out_dim))
            # Batch Normalization Layer
            self.bn_layers.append(BatchNorm(current_out_dim, momentum=0.99))
            # ReLU Activation Layer
            self.relu_layers.append(nn.PReLU())
            # Update the input dimension for the next layer
            current_in_dim = current_out_dim
            
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = self.relu_layers[i](x)
        return x
    
            
        

class ContrastiveGNN(nn.Module):
    def __init__(self, config, input_dim, num_classes):
        """
        Initialize the ContrastiveGNN model.
        """
        super(ContrastiveGNN, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.num_classes = int(num_classes)
        self.num_layers = config['model']['num_layers']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['output_dim']
        self.gnn_block = GNNBlock(config, input_dim, self.hidden_dim, self.output_dim, self.num_layers)
        self.device = config['device']
        self.tau = config['hyperparameters']['tau']
        self.thres = config['hyperparameters']['thres']
        # build two layers of classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.num_classes)
        )
        
    def _drop_edge(self, data, p=0.5):
        """
        Drop edges from the graph by randomly setting some of the edge weights to zero.
        """
        edge_index = data.edge_index
        edge_index = dropout_adj(edge_index, p=p)
        data.edge_index = edge_index[0]
        return data
    
    def _mask_node(self, data, p=0.5):
        """
        Mask the nodes of the graph by replacing some of the node features with the mean of the node features.
        """
        x = data.x  
        tensor_mean = torch.ones_like(x) * x.mean(dim=0)
        # tensor_mean = torch.zeros_like(x)
        
        # generate a mask with the same shape as x, filled with 0 and 1
        mask = torch.rand(x.size()) < p
        # replace the masked indices with the mean
        x[mask] = tensor_mean[mask]
        data.x = x
        return data
    
    
    def transform(self, data, p_edge=0.5, p_node=0.5):
        # make a copy of x and edge_index
        data_copy = deepcopy(data)
        # apply the transformation
        data_copy = self._mask_node(data_copy, p_node)
        data_copy = self._drop_edge(data_copy, p_edge)
        return data_copy
    
    def _soft_nn(self, query, supports, labels):
        """
        Computes the soft nearest neighbor (SNN) probabilities.

        This method normalizes the query and support embeddings, computes the similarity 
        between the query and support embeddings using a softmax function, and then 
        applies the resulting probabilities to the provided labels.

        Args:
            query (torch.Tensor): The query embeddings to be compared against the supports.
            supports (torch.Tensor): The support embeddings used for comparison.
            labels (torch.Tensor): The ground truth labels corresponding to the supports.

        Returns:
            torch.Tensor: The computed probabilities for the query embeddings based on the 
                          soft nearest neighbor approach.
        """
        query = F.normalize(query)
        supports = F.normalize(supports)

        return F.softmax(query @ supports.T / self.tau, dim=1) @ labels

    def label_consistency_loss(self, anchor, positive, anchor_sup, positive_sup, labels, gt_labels, train_mask):
        """
        Compute the label consistency loss between anchor and positive embeddings.
        """
        gt_labels = torch.tensor(gt_labels)
        train_mask = train_mask.to(self.device)
        with torch.no_grad():
            gt_labels = gt_labels[train_mask].unsqueeze(-1).reshape(-1, 1)
            # convert gt_labels to int64
            gt_labels = gt_labels.to(torch.int64)
            matrix = torch.zeros(train_mask.sum().item(), self.num_classes).to(self.device)
            
            gt_matrix = matrix.scatter_(1, gt_labels, 1)

        probs1 = self._soft_nn(anchor, anchor_sup, labels)
        with torch.no_grad():
            targets1 = self._soft_nn(positive, positive_sup, labels)
            values, _ = targets1.max(dim=1)
            boolean = torch.logical_or(values>self.thres, train_mask)
            indices1 = torch.arange(len(targets1)).to(self.device)[boolean]
            targets1[targets1 < 1e-4] *= 0
            # print(targets1.shape, gt_matrix.shape)
            targets1[train_mask] = gt_matrix            
            targets1 = targets1[indices1]

        probs1 = probs1[indices1]
        loss = torch.mean(torch.sum(torch.log(probs1**(-targets1)), dim=1))

        return loss
    
    def similarity_loss(self, anchor_embeddings, positive_embeddings):
        """
        Computes the similarity loss between anchor and positive embeddings using cosine similarity.

        This method calculates the cosine similarity between the anchor and positive embeddings,
        and then derives a loss value based on the mean cosine similarity. The loss is designed
        to encourage the anchor and positive embeddings to be close to each other in the embedding
        space.

        Args:
            anchor_embeddings (torch.Tensor): The embeddings of the anchor samples.
            positive_embeddings (torch.Tensor): The embeddings of the positive samples.

        Returns:
            torch.Tensor: The computed similarity loss, which is minimized when the anchor and 
                          positive embeddings are similar.
        """
        cosine_similarity = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=-1)
        return 2 - 2 * cosine_similarity.mean()
    
    def supervised_loss(self, anchor_prediction, positive_prediction, labels):
        """
        Compute the supervised loss between anchor and positive predictions.
        """
        return (F.cross_entropy(anchor_prediction, labels) + F.cross_entropy(positive_prediction, labels)) / 2
    
    def forward(self, data):
        """
        Forward pass of the model.
        """
        embeddings = self.gnn_block(data.x, data.edge_index)
        logits = self.classification_head(embeddings)
        out = torch.argmax(logits, dim=1)
        return embeddings, logits, out
