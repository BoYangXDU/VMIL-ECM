"""
@Author: Bo Yang
@Organization: School of Artificial Intelligence, Xidian University
@Email: bond.yang@outlook.com
@LastEditTime: 2024-07-31
"""

from utils.loaddata import loadmat
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset
from typing import List, Tuple

class customDataset(Dataset):

    def __init__(self, samples: List[Tuple]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

#train
def train_data_cause(path: str):
    data = loadmat(path)
    dataBags = data['dataBags']
    labelsB = data['labelsB']
    train_data = data['X']
    num_pos_bags = data['params']['num_pbags']
    num_neg_bags = data['params']['num_nbags']
    dataBags_format = []
    for i in range(num_pos_bags+num_neg_bags):
        dataBags_format.append(dataBags[i])
    dataBags_format_array = np.array(dataBags_format)
    inputs = list(zip(dataBags_format_array, labelsB))
    train_dataset = customDataset(inputs)
    return train_dataset


# validation
def val_data_cause(path: str):
    data = loadmat(path)
    GTlabels_point = data['GTlabels_point']
    train_data = data['X']
    train_data = train_data.T
    inputs = list(zip(train_data, GTlabels_point))
    val_dataset = customDataset(inputs)
    return val_dataset

# test
def tes_data_cause(path: str):
    data = loadmat(path)
    GTlabels_point = data['GTlabels_point']
    train_data = data['X']
    train_data = train_data.T
    inputs = list(zip(train_data, GTlabels_point))
    test_dataset = customDataset(inputs)
    return test_dataset

