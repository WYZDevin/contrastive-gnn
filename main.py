

from utils import get_xgboost_result, set_random_seeds

import yaml
import torch
from train import train_constrastive
from model import ContrastiveGNN
from data import Dataset
from torch.utils.tensorboard import SummaryWriter
import os

def main():

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    set_random_seeds(0)
    # The ratio for the (training, validation). A choice between (0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2), (0.6, 0.2)
    ratio = (0.6, 0.2)
    # The fold. A choice between 0, 1, 2, 3, 4
    fold = 3
    
    data = Dataset(root=config["data"]["root"], dataset=config["data"]["dataset"], config=config)
    data.update_mask(fold=fold, ratio=ratio)
    
    # get benchmark result for xgboost  
    xgboost_result = get_xgboost_result(data, ratio=ratio, fold=fold)
    print(f'XGBoost result: Best Val Acc: {xgboost_result["validation_acc"]} - Best Test Acc: {xgboost_result["test_acc"]}')
    
    # get benchmark result for contrastive gnn
    model = ContrastiveGNN(config, data.x.shape[1], data.y.max().item() + 1).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr"], weight_decay=config["pretrain"]["weight_decay"])
    writer = SummaryWriter(log_dir=os.path.join(config["dir_log"], "pretrain"))
    train_constrastive(config, model, optimizer, data, writer)

if __name__ == "__main__":
    main()