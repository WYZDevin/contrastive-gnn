from copy import deepcopy
from data import Dataset
from model import ContrastiveGNN
from sampling import Sampler
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from utils import get_xgboost_result

def evaluate(model, data):
    """
    Evaluate the model on the given data.
    """
    model.eval()
    embeddings, logits, out = model(data)
    train_pred, train_labels = out[data.train_mask], data.y[data.train_mask].reshape(-1)
    val_pred, val_labels = out[data.val_mask], data.y[data.val_mask].reshape(-1)
    test_pred, test_labels = out[data.test_mask], data.y[data.test_mask].reshape(-1)
    train_acc = torch.sum(train_pred == train_labels).float() / len(train_labels)
    val_acc = torch.sum(val_pred == val_labels).float() / len(val_labels)
    test_acc = torch.sum(test_pred == test_labels).float() / len(test_labels)
    
    return train_acc, val_acc, test_acc

def train_constrastive(config, model, optimizer, data, writer=None):
    """
    Train the model using contrastive learning.
    """
    model.train()
    data = data.to(config["device"])
    p_edge_1 = config['augmentation']['p_edge_1']
    p_node_1 = config['augmentation']['p_node_1']
    p_edge_2 = config['augmentation']['p_edge_2']
    p_node_2 = config['augmentation']['p_node_2']
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    
    labels = deepcopy(data.y).squeeze(-1).to(torch.int64)
    sampler = Sampler(data, labels, train_mask)
    
    dict_best_result = {
        'validation_acc': 0,
        'test_acc': 0,
        'best_epoch': 0,
    }
    
    for epoch in range(config['pretrain']['num_epochs']):
        optimizer.zero_grad()
        
        # conduct strong and weak augmentation
        anchor, positive = model.transform(data, p_edge_1, p_node_1), model.transform(data, p_edge_2, p_node_2)
        
        # sample the nodes for the contrastive learning
        label_matrix, support_index = sampler.sample()
        label_matrix = label_matrix.to(config["device"])
        
        # get the embeddings of the anchor and positive nodes
        anchor_embeddings, anchor_logits, _ = model(anchor)
        positive_embeddings, positive_logits, _ = model(positive)
        anchor_sup_embeddings = anchor_embeddings[support_index]
        positive_sup_embeddings = positive_embeddings[support_index]
        
        # compute the consistency loss  
        consistency_loss = model.label_consistency_loss(anchor_embeddings, positive_embeddings, anchor_sup_embeddings, positive_sup_embeddings, label_matrix, labels, train_mask)
        
        # compute the similarity loss
        similarity_loss = model.similarity_loss(anchor_embeddings[train_mask], positive_embeddings[train_mask])
        
        # compute the supervised loss
        supervised_loss = model.supervised_loss(anchor_logits[train_mask], positive_logits[train_mask], labels[train_mask])
        
        # compute the total loss
        lambda_1 = config['hyperparameters']['lam']
        lambda_2 = config['hyperparameters']['lam2']
        loss = lambda_1 * consistency_loss + lambda_2 * similarity_loss + supervised_loss
        
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = evaluate(model, data)
        if val_acc > dict_best_result['validation_acc']:
            dict_best_result['validation_acc'] = val_acc
            dict_best_result['test_acc'] = test_acc
            dict_best_result['best_epoch'] = epoch
            
        log_msg = f"Epoch {epoch+1}/{config['pretrain']['num_epochs']} - "
        log_msg += f"Loss: {loss:.4f} - "
        log_msg += f"Val Acc: {val_acc:.4f} - "
        log_msg += f"Test Acc: {test_acc:.4f}"
        log_msg += f"Best Epoch: {dict_best_result['best_epoch']}"
        log_msg += f"Best Val Acc: {dict_best_result['validation_acc']:.4f} - "
        log_msg += f"Best Test Acc: {dict_best_result['test_acc']:.4f}"
        # print(log_msg, end='\r')
        
        # update tensorboard
        if writer is not None:
            writer.add_scalar('Consistency Loss/train', consistency_loss, epoch)
            writer.add_scalar('Similarity Loss/train', similarity_loss, epoch)
            writer.add_scalar('Supervised Loss/train', supervised_loss, epoch)
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
    print(f"Contrastive Learning: Best Val Acc: {dict_best_result['validation_acc']:.4f} - Best Test Acc: {dict_best_result['test_acc']:.4f}")
    return dict_best_result


def train_supervised(config, model, optimizer, data, writer=None, ratio=(0.2, 0.2), fold=1):
    """
    Train the model using supervised learning.
    This is basically the same as the contrastive learning, but without the augmentation.
    """
    model.train()
    data = data.to(config["device"])
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    
    labels = deepcopy(data.y).squeeze(-1).to(torch.int64)
    
    dict_best_result = {
        'validation_acc': 0,
        'test_acc': 0,
        'best_epoch': 0,
    }
    for epoch in range(config['pretrain']['num_epochs']):
        optimizer.zero_grad()
        
        embeddings, logits, out = model(data)
        
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        train_acc, val_acc, test_acc = evaluate(model, data)
        log_msg = f"Epoch {epoch+1}/{config['pretrain']['num_epochs']} - "
        log_msg += f"Loss: {loss:.4f} - "
        log_msg += f"Val Acc: {val_acc:.4f} - "
        log_msg += f"Test Acc: {test_acc:.4f}"
        # print(log_msg, end='\r')
        
        if val_acc > dict_best_result['validation_acc']:
            dict_best_result['validation_acc'] = val_acc
            dict_best_result['test_acc'] = test_acc
            dict_best_result['best_epoch'] = epoch
    print(f"Supervised Learning: Best Val Acc: {dict_best_result['validation_acc']:.4f} - Best Test Acc: {dict_best_result['test_acc']:.4f}")
    return dict_best_result


def train_one_set(config, ratio, fold):
    """
    Train the model for one set of ratio and fold settings.
    """
    data = Dataset(root=config["data"]["root"], dataset=config["data"]["dataset"], config=config)
    # update the mask for the train, val, and test nodes
    data.update_mask(fold=fold, ratio=ratio)
    # get the benchmark result for xgboost
    xgboost_result = get_xgboost_result(data, ratio=ratio, fold=fold)
    
    # train the model using contrastive learning
    model = ContrastiveGNN(config, data.x.shape[1], data.y.max().item() + 1).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr"], weight_decay=config["pretrain"]["weight_decay"])
    constrative_result = train_constrastive(config, model, optimizer, data)
    
    # train the model using supervised learning
    model = ContrastiveGNN(config, data.x.shape[1], data.y.max().item() + 1).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr"], weight_decay=config["pretrain"]["weight_decay"])
    supervised_result = train_supervised(config, model, optimizer, data)
    
    # get the final result
    dict_final_result = {
        'xgb_val_acc': xgboost_result['validation_acc'],
        'xgb_test_acc': xgboost_result['test_acc'],
        'constrative_val_acc': constrative_result['validation_acc'],
        'constrative_test_acc': constrative_result['test_acc'],
        'supervised_val_acc': supervised_result['validation_acc'],
        'supervised_test_acc': supervised_result['test_acc'],
    }
    
    return dict_final_result