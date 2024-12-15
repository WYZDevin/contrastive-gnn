

import pickle
import torch
import os
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon

from noise_graph_data import NoiseDataGraph 


class Dataset(InMemoryDataset):

    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, root="data", dataset='noise', transform=None, pre_transform=None, config=None):
        """
        Initializes the Dataset class for multi-view dataset creation through graph data augmentation.

        Parameters:
        - root (str): The root directory where the dataset will be stored.
        - dataset (str): The name of the dataset to be used (e.g., 'noise').
        - transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
        - pre_transform (callable, optional): A function/transform that takes in a Data object and returns a pre-transformed version.
        - config (dict): Configuration parameters for the dataset, including augmentation settings and model parameters.

        This constructor creates the necessary directories for storing the dataset, loads the processed data from disk,
        and initializes the parent InMemoryDataset class with the specified parameters.
        """
        self.config = config
        self.dataset = dataset
        self.data_dir = osp.join(root, dataset)
        os.makedirs(self.data_dir, exist_ok=True)
        super().__init__(root=self.data_dir, transform=transform, pre_transform=pre_transform)
        path = osp.join(self.processed_dir, self.processed_file_names[0])
        self.data, self.slices = torch.load(path)
        

    def process_full_batch_data(self, data):
        """
        Process the full batch data for the noise dataset.
        """
        print("Processing full batch data")
        G = pickle.load(open('/home/dd/storage/data-storage/gnn_noise_model/processed_geodata/noise_graph_6nn.pkl', 'rb'))
        _nois_data = NoiseDataGraph(root=self.root, config=self.config, G=G, image_dir=self.config["dir_image"])
        data = _nois_data.preprocess_data(include_tabular=self.config["model"]["include_tabular"], 
                                   include_image=self.config["model"]["include_image"], 
                                   label_type="noise")

        return [data]

    def process(self):
        """
        Process either a full batch or cluster data.

        :return:
        """
        processed_path = osp.join(self.processed_dir, self.processed_file_names[0])
        if self.dataset == 'noise':
            data = self.read_noise_data()
            data, slices = self.collate([data])
            torch.save((data, slices), processed_path)
            return
            
        
        if not osp.exists(processed_path):
            path = osp.join(self.raw_dir, self.raw_file_names[0])
            data, _ = torch.load(path)
            data_list = self.process_full_batch_data(data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)
            
    def read_noise_data(self):
        """
        Read the noise data from the processed GeoData file.
        """
        G = pickle.load(open('PATH', 'rb'))
        _nois_data = NoiseDataGraph(root=self.data_dir, config=self.config, G=G, image_dir=self.config["dir_image"])
        data = _nois_data.preprocess_data(include_tabular=self.config["model"]["include_tabular"], 
                                   include_image=self.config["model"]["include_image"], 
                                   label_type="noise", random_seed=0)
        # create masks for train, val, test
        return data
    
    def update_mask(self, fold, ratio):
        """
        Given the train/test ratio and fold, update the train, validation, and test masks for the dataset.
        """
        self.train_mask = self.data.dict_masks[ratio][fold]["train_mask"]
        self.val_mask = self.data.dict_masks[ratio][fold]["val_mask"]
        self.test_mask = self.data.dict_masks[ratio][fold]["test_mask"]
    
    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        return [f'byg.data.pt']

    @property
    def raw_dir(self):
        return osp.join(self.data_dir, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.data_dir, "processed")

    @property
    def dirs(self):
        return [self.raw_dir, self.processed_dir]

    def download(self):
        pass

