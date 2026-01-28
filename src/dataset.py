"""
@File   :   dataset.py
@Desc   :   Constructor of datasets
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import *
import pickle

class MorphDataset(Dataset):
    def __init__(self, CP_data):
        self.CP_data = CP_data
        # Metadata_Well
        self.CP_index = CP_data['CP_index']
        # Metadata_JCP2022
        self.mol_id = CP_data['Metadata_JCP2022']
        # Features
        self.x1 = CP_data['x1'].astype(np.float32)
        self.x2 = CP_data['x2'].astype(np.float32)
        self.mol_fp = load_pickle('../Dataset/Train_compounds.pkl')

    def __getitem__(self, index):
        mol_feature = self.mol_fp[self.mol_id[index]]
        return self.x1[index], self.x2[index], mol_feature, self.mol_id[index], self.CP_index[index]

    def __len__(self):
        return self.mol_id.shape[0]

class MorphDataset_prediction(Dataset):

    def __init__(self, x1, mol_feature, mol_id):
        x1_data = []
        x1 = x1[0]
        # If the data passed to x1 is 2-dimensional, it needs to be processed.
        if np.ndim(x1) == 1 and len(mol_id) != 1:
            # If the default DMSO data is used, there will only be one row, so initialize according to the row number of mol_id
            for i in range(len(mol_id)):
                x1_data.append(x1)
        elif np.ndim(x1) == 1 and len(mol_id) == 1:
            x1_data = np.array([x1])
        else:
            x1_data = np.array(x1)
        mol_feature_data = []
        for i, row in enumerate(mol_id):
            mol_feature_data.append(mol_feature[row])
        # Control data
        self.x1 = np.asarray(x1_data)
        # Metadata
        self.mol_feature = np.asarray(mol_feature_data)
        # CP_id
        self.mol_id = np.asarray(mol_id)

    def __getitem__(self, index):
        return self.x1[index], self.mol_feature[index], self.mol_id[index]

    def __len__(self):
        return self.mol_feature.shape[0]

class Dataset_x1(Dataset):
    def __init__(self, CP_data):
        data = process_feature(CP_data, 'x1')

        self.CP_data = data
        # Metadata_Well
        self.CP_index = data['CP_index']
        # Metadata_JCP2022
        self.mol_id = data['Metadata_JCP2022']
        # Feature name
        self.feature = data['train_feat']
        # Control data
        self.x1 = data['x1']

    def __getitem__(self, index):
        # Returns x1 feature data, CP_index, and mol_id
        return self.x1[index], self.CP_index[index], self.mol_id[index]

    def __len__(self):
        return self.CP_index.shape[0]


class Dataset_x2(Dataset):

    def __init__(self, CP_data):
        data = process_feature(CP_data, 'x2')

        self.CP_data = data
        # Metadata_Well
        self.CP_index = data['CP_index']
        # Metadata_JCP2022
        self.mol_id = data['Metadata_JCP2022']
        # Feature name
        self.feature = data['train_feat']
        # Treatment data
        self.x2 = data['x2']

    def __getitem__(self, index):
        # Returns x2 feature data, CP_index, and mol_id
        return self.x2[index], self.CP_index[index], self.mol_id[index]

    def __len__(self):
        return self.CP_index.shape[0]