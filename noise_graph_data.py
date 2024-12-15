
# Define the custom dataset class that handles data preprocessing
import logging
import torch
import os

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split

from constant_variables import FEATURES

class GraphImageDataset(Dataset):
    def __init__(self, image_dir, node2image):
        self.image_dir = image_dir
        self.node2image = node2image
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.node2image)

    # @lru_cache(maxsize=50000)
    # def __getitem__(self, idx):
    #     # get a single image
    #     img_path = os.path.join(self.image_dir, self.node2image[idx][0])
    #     img = Image.open(img_path).convert('RGB')
    #     if self.transform:
    #         img = self.transform(img)
    #     return img
    
    def __getitem__(self, idx):
        # now, return (4, 3, 224, 224)
        imgs = []
        for i in range(4):
            img_path = os.path.join(self.image_dir, self.node2image[idx][i])
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        return torch.stack(imgs)

class NoiseDataGraph(InMemoryDataset):
    """
    Custom dataset class that handles data preprocessing.
    The class converts the networkx graph into a PyTorch Geometric Data object.
    """
    # initialize the road class map

    def __init__(self, root, G, config, noise_col='noise', transform=None, pre_transform=None, image_dir=None):
        self.road_class_map = {
            'Local / Street' : 0,
            'Local / Strata' : 0,
            'Collector' : 1,
            'Arterial' : 2,
            'Service' : -1,
            'Rapid Transit' : -1,
            'Ramp' : 3,
            'Freeway' : 3
        }
        self.config = config
        
        self.G = G  # The graph structure
        self.y_noise_id = {node_id : G.nodes[node_id]['noise'] for node_id in G.nodes if type(G.nodes[node_id]['noise']) == list and len(G.nodes[node_id]['noise']) > 0}
        # ground truth labels
        self.y_noise = {node_id: G.nodes[node_id]['noise_class'] for node_id in G.nodes}
        
        # semi-supervised label 1
        self.y_road = {node_id: self.road_class_map[G.nodes[node_id]['ROAD_CLASS']] for node_id in G.nodes}
        
        # semi-supervised label 2, use quantile to put them into 4 bins
        # all_psudo_noise = np.array([G.nodes[i][f'psudo_noise{config["fold"]}'] for i in range(len(G.nodes))])
        all_psudo_noise = np.array([G.nodes[i]['psudo_noise'] for i in range(len(G.nodes))])
        # qcut = pd.qcut(all_psudo_noise, config['pretrain']['out_channels'], labels = list(range(config['pretrain']['out_channels'])))
        qcut = pd.cut(all_psudo_noise, 4, labels = (0, 1, 2, 3), ordered=False)

        self.y_psudo = {node_id: qcut[node_id]  for node_id in G.nodes}
        
        self.feature_names = FEATURES
        self.noise_col = noise_col  # Column to exclude from features
        self.image_dir = image_dir

        # initialize the ogfid2image dictionary
        self.ogfid2node = {}
        for node in self.G.nodes:
            self.ogfid2node[int(self.G.nodes[node]['OGF_ID'])] = node
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.ogfid2image = {}
        lst_filenames = os.listdir(image_dir)
        for filename in lst_filenames:
            ogfid = int(filename.split('_')[0])
            if ogfid not in self.ogfid2image:
                self.ogfid2image[ogfid] = []
            self.ogfid2image[ogfid].append(filename)

        # create a node2image dictionary for extracting images
        self.node2image = {}
        
        for node in self.G.nodes:
            # read and transform the image
            self.node2image[node] = self.ogfid2image[int(self.G.nodes[node]['OGF_ID'])]
        self.images = {}
        super().__init__(root, transform, pre_transform)
        logging.info("Starting data preprocessing...")


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        pass


    def preprocess_data(self, include_tabular=True, include_image=True, label_type='', random_seed=0):
        # create a graph dataset basedo n settings
        # label_type: 'noise' | road_class' | 'psudo_noise'
        # Create the [n_node, n_feature] tensor

        feature_names = []
        if include_tabular:
            feature_names = self.feature_names
        
        node_features = torch.zeros((len(self.G.nodes), len(feature_names)))
        logging.info("Creating node features tensor...")
        for node in self.G.nodes:
            for j, feature_name in enumerate(feature_names):
                node_features[node, j] = self.G.nodes[node][feature_name]

        # normalize the node features
        node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)
        
        if include_image:
            # # make a tensor to store the node id in int64
            node_ids = torch.arange(len(self.G.nodes), dtype=torch.int64)
            node_features = torch.cat([node_ids.unsqueeze(1), node_features], dim=1)
            # load the image features
            # node_features = torch.cat([node_features, torch.zeros((len(self.G.nodes), 4*3*224*224))], dim=1)
            # for node in self.G.nodes:
            #     node_features[node, -4*3*224*224:] = self.images[node].flatten()

        # Define the edge index (2, 2 * n_edges) for a bidirectional graph
        edge_index = torch.zeros((2, 2 * len(self.G.edges)), dtype=torch.long)
        # construct edge attributes
        # edge_attr = torch.zeros((2 * len(self.G.edges), 1))
        logging.info("Defining edge index...")
        
        for i, edge in enumerate(self.G.edges):
            edge_index[0, 2 * i] = edge[0]
            edge_index[1, 2 * i] = edge[1]
            # edge_attr[2 * i, 0] = self.G.edges[edge]['distance']
            edge_index[0, 2 * i + 1] = edge[1]
            edge_index[1, 2 * i + 1] = edge[0]
            # edge_attr[2 * i + 1, 0] = self.G.edges[edge]['distance']


        
        y_tensor = torch.zeros((len(self.G.nodes),1), dtype=torch.float32)
        for node in self.G.nodes:
            
            if node in self.y_noise:
                y_tensor[node, 0] = self.y_noise[node]
            else:
                y_tensor[node, 0] = -1

        data = Data(x=node_features, edge_index=edge_index, y=y_tensor)
        self.node_features = node_features

        # split the mask into train, val, test for the node that the noise is not -1
        avaliable_data = [node for node in self.G.nodes if node in self.y_noise_id]
        avaliable_data_indices = torch.tensor(avaliable_data, dtype=torch.int64)
           
        avaliable_data_mask = torch.zeros(len(self.G.nodes), dtype=torch.bool)
        avaliable_data_mask[avaliable_data_indices] = True

        data.avaliable_data_mask = avaliable_data_mask

        # shuffle the avaliable data with torch
        # shuffled_avaliable_data_indices = torch.randperm(len(avaliable_data_indices))
        # set the random seed for nump
        ratios = [(0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2), (0.6, 0.2)]
        dict_masks = {}
        for ratio in ratios:
            dict_masks[ratio] = []
            for i in range(self.config["folds"]): 
                np.random.seed(i)
                current_mask = {}
                train_ratio, val_ratio = ratio
                shuffled_avaliable_data_indices = np.random.permutation(avaliable_data_indices)

                train_indices = shuffled_avaliable_data_indices[:int(len(avaliable_data_indices)*train_ratio)]    
                val_indices = shuffled_avaliable_data_indices[int(len(avaliable_data_indices)*train_ratio):int(len(avaliable_data_indices)*(train_ratio+val_ratio))]

                # test_indices = shuffled_avaliable_data_indices[int(len(avaliable_data_indices)*(self.config["data"]["train_ratio"]+self.config["data"]["val_ratio"])):]
                # use remaining node for test
                # test_indices = [node for node in self.G.nodes if node not in train_indices and node not in val_indices]
                test_indices = shuffled_avaliable_data_indices[int(len(avaliable_data_indices)*(train_ratio+val_ratio)):]
                train_mask = torch.zeros(len(self.G.nodes), dtype=torch.bool)
                train_mask[train_indices] = True
                val_mask = torch.zeros(len(self.G.nodes), dtype=torch.bool)
                val_mask[val_indices] = True
                test_mask = torch.zeros(len(self.G.nodes), dtype=torch.bool)
                test_mask[test_indices] = True

                current_mask["train_mask"] = train_mask
                current_mask["val_mask"] = val_mask
                current_mask["test_mask"] = test_mask
                dict_masks[ratio].append(current_mask)
        # data.G = self.G
        data.dict_masks = dict_masks
        # validate the data
        print(data.y[train_mask])
        # # Build the y_tensors for each label type
        # y_tensors = []
        # logging.info("Building y_tensors for each label type...")
        # for i in range(len(list(self.y.values())[0])):
        #     y_tensor = torch.zeros((len(self.G.nodes), 1))
        #     for j, node in enumerate(self.G.nodes):
        #         if node in self.y:
        #             y_tensor[j, 0] = self.y[node][i]
        #         else:
        #             y_tensor[j, 0] = -1
        #     y_tensors.append(y_tensor)

        # Create Data objects for each label type
        # data_list = []
        # for i in range(len(y_tensors)):
        #     data = Data(x=node_features, edge_index=edge_index, y=y_tensors[i])
        #     data_list.append(data)

        # self.data, self.slices = self.collate([data])
        return data
