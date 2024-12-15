# Contrasitive Graph Neural Networks for Node Classification

This project implements contrastive learning approaches for GNN to improve node classification performance in small datasets. The implementation is based on the paper "[Contrastive Learning for Graph Neural Networks](https://github.com/Junseok0207/GraFN)". Some of the code is adapted from the original code of the paper, but majority of the impementation was written by our group.

The data we are using is a provided dataset collected in collaboration with the local government, therefore we are unable to release the dataset. However, if needed, we can provide a mock dataset with random noise added to the original data for testing (since the original data contains sensitive information), if Prof/TA is interested in running the pipeline.

## Overview

The project compares three different approaches for node classification:
- XGBoost (baseline)
- Supervised GNN learning 
- Contrastive GNN learning with data augmentation

The contrastive learning approach uses both weak and strong augmentations of the input graph to create positive pairs for contrastive learning, while maintaining label consistency.

## Requirements

- PyTorch
- PyTorch Geometric
- XGBoost
- scikit-learn
- tensorboard

All the dependencies can be installed by running the 'pip install -r requirements.txt' command.

## Usage

1. Run the 'main.py' would directly show the result of the XGBoost and Contrastive GNN. The output of the Contrastive GNN may varies due to the randomness of the data augmentation.

2. 'TestMoelPerformance.ipynb' is used to test the performance of the model with different settings of the ratio and fold. It shows how we run the model for all the settings and compare the result, and the hyperparameter tuning process.


