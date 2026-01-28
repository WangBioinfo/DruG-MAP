"""
@File   :   utils.py
@Desc   :   Utility functions for model training
"""
import os
import torch
import numpy as np
import pandas as pd
import sklearn
import random
import h5py
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging
import sys

# Load H5 data based on file path
def load_HDF5(filename):
    h5_data = {}
    with h5py.File(filename, 'r') as f:
        for key in f:
            # print("------------key: %s--------------------",key)
            h5_data[key] = np.asarray(f[key])
            if isinstance(h5_data[key][0], np.bytes_):
                h5_data[key] = h5_data[key].astype(str)
    return h5_data

# Save the data to an H5 file
def save_HDF5(filename, data):
    with h5py.File(filename, 'w') as f:
        for key, item in data.items():
            if isinstance(item[0], str):
                item = item.astype(np.bytes_)
            f[key] = item

# Load Parquet file
def load_parquet(filename):
    par_data = pd.read_parquet(filename)
    return par_data

# Load pickle file
def load_pickle(filename):
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result

# Save pickle file
def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

# Set random seed
def setup_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_metric_func(metric):
    # RMSE
    if metric == 'rmse':
        return rmse
    # MAE
    if metric == 'mae':
        return mean_absolute_error
    # R2
    if metric == 'r2':
        return r2_score
    # pearson
    if metric == 'pearson':
        return pearson
    # spearman
    if metric == 'spearman':
        return spearman

    raise ValueError(f'Metric "{metric}" not supported.')


def rmse(targets, preds):
    return np.sqrt(mean_squared_error(targets, preds))

def pearson(targets, preds):
    try:
        return pearsonr(targets, preds)[0]
    except ValueError:
        print(targets, preds)
        print(np.isnan(targets), np.isnan(preds))
        return float('nan')

def spearman(targets, preds):
    try:
        return spearmanr(targets, preds)[0]
    except ValueError:
        return float('nan')

def subsetDict(d, ind):
    # print("====subsetDict======")
    res = {}
    if not isinstance(ind, np.ndarray):
        ind = np.asarray(ind)
    for k in d.keys():
        if isinstance(d[k], np.ndarray):
            res[k] = d[k][ind]
    return res

# Returns the index of the value val in arr.
def get_index_by_value(arr, val):
    return list(filter(lambda x: arr[x] == val, range(len(arr))))

def getSplitsByGroupKFold(groups, n_splits, shuffle, random_state):
    print("====getSplitsByGroupKFold======")
    assert (n_splits >= 3)
    kf = GroupKFold(n_splits=n_splits)
    if shuffle:
        unique_groups = np.unique(groups)
        rnd_renames = sklearn.utils.shuffle(np.arange(len(unique_groups)), random_state=random_state)
        groups_renamed = np.array([rnd_renames[np.argwhere(unique_groups == g)[0]][0] for g in groups])
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups_renamed)
    else:
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups)

    folds = [list(x[1]) for x in kfsplit]
    folds_nums = list(range(len(folds)))

    tr_fold_nums = folds_nums[:-2]
    ind_tr = sum([folds[i] for i in tr_fold_nums], [])
    ind_va = folds[folds_nums[-2]]
    ind_te = folds[folds_nums[-1]]

    return ind_tr, ind_va, ind_te

def getSplitsByStratifiedKFold(groups, n_splits, shuffle, random_state):
    print("====getSplitsByStratifiedKFold======")
    assert (n_splits >= 3)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    if shuffle:
        unique_groups = np.unique(groups)
        rnd_renames = sklearn.utils.shuffle(np.arange(len(unique_groups)), random_state=random_state)
        groups_renamed = np.array([rnd_renames[np.argwhere(unique_groups == g)[0]][0] for g in groups])
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), y=groups_renamed)
    else:
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), y=groups)

    folds = [list(x[1]) for x in kfsplit]
    folds_nums = list(range(len(folds)))

    tr_fold_nums = folds_nums[:-2]
    ind_tr = sum([folds[i] for i in tr_fold_nums], [])
    ind_va = folds[folds_nums[-2]]
    ind_te = folds[folds_nums[-1]]

    return ind_tr, ind_va, ind_te

def split_data(data, n_folds=5, split_type='random_split', rnds=None):
    print("====split_data======")
    if split_type == 'random_split':
        idx_train, idx_valid, idx_test = getSplitsByGroupKFold(data['CP_index'], n_folds, shuffle=True, random_state=rnds)
    elif split_type == 'smiles_split':
        idx_train, idx_valid, idx_test = getSplitsByStratifiedKFold(data['mol_id'], n_folds, shuffle=True, random_state=rnds)
    ppair = subsetDict(data, idx_train)
    ppairv = subsetDict(data, idx_valid)
    ppairt = subsetDict(data, idx_test)

    return ppair, ppairv, ppairt

def check_path_exists(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('Directory created successfully')
    else:
        print('Directory already exists')

def process_feature(CP_data, feat_key):
    features = pd.read_csv("../Dataset/features_with_type.csv")
    # t_feat = features[features['train_flag'] == 1]['feature_name'].tolist()
    t_feat = features[features['train_flag'] == 1]['harmony'].tolist()
    # 加载预训练数据
    keys = CP_data.keys()
    data = {}
    x = []
    train_feat = []
    for key in keys:
        if key.startswith('Metadata_'):
            data[key] = np.asarray(CP_data[key])
        elif key in t_feat:
            x.append(CP_data[key])
            train_feat.append(key)

    if feat_key in CP_data.keys():
        data[feat_key] = np.asarray(CP_data[feat_key])
    else:
        data[feat_key] = np.transpose(x)
    data['train_feat'] = np.asarray(train_feat)
    data['CP_index'] = np.asarray(CP_data['CP_index'])
    data['mol_id'] = np.asarray(CP_data['mol_id'])
    return data

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_logger(name, flag):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s]: %(message)s')
    if flag == 'file':
        fh = logging.FileHandler(f'./logs/{name}.log', 'a', 'utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        st = logging.StreamHandler(sys.stdout)
        st.setFormatter(formatter)
        logger.addHandler(st)
    return logger