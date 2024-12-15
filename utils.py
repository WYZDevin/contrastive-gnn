import torch
import numpy as np
import random
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def set_random_seeds(random_seed=0):
    """
    Set random seeds for reproducibility for all modules.
    """
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    

def get_xgboost_result(data, ratio, fold):
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    X_train, y_train = data.x[train_mask], data.y[train_mask]
    X_val, y_val = data.x[val_mask], data.y[val_mask]
    X_test, y_test = data.x[test_mask], data.y[test_mask]

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    return {
        'validation_acc': val_acc,
        'test_acc': test_acc,
    }