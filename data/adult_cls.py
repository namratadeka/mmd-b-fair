import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.adult_process import get_adult_data

class AdultClsDataset(Dataset):
    """
    The UCI Adult dataset.
    """

    def __init__(self, phase, tar_attr='income', priv_attr='sex', clr_ratio=None, train_path='', test_path=''):
        self.tar_attr = tar_attr
        self.priv_attr = priv_attr
        if phase == 'test':
            clr_ratio = [0.5, 0.5]

        self.data = get_adult_data(tar_attr, priv_attr, clr_ratio, train_path, test_path)
        if phase not in ["train", "val", "test"]:
            raise NotImplementedError

        if phase == "train":
            self.X = self.data[f"x_train"][self.data["train_inds"]]
            self.Y = self.data[f"y_train"][self.data["train_inds"]]
            self.A = self.data[f"attr_train"][self.data["train_inds"]]
        elif phase == "val":
            self.X = self.data[f"x_train"][self.data["valid_inds"]]
            self.Y = self.data[f"y_train"][self.data["valid_inds"]]
            self.A = self.data[f"attr_train"][self.data["valid_inds"]]
        elif phase == "test":
            self.X = self.data[f"x_test"]
            self.Y = self.data[f"y_test"]
            self.A = self.data[f"attr_test"]
        else:
            raise Exception("Wrong phase")

        self.input_shape = self.X.shape
        self.num_samples = self.input_shape[0]
        self.xdim = self.X.shape[1]
        self.ydim = 1
        self.adim = 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.tensor(self.Y[idx]).long(),
            torch.tensor(self.A[idx]).long()
        )

class AdultClsBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, mode, tar_attr, priv_attr, clr_ratio, train_path, test_path, **_ignored):

        self._instance = AdultClsDataset(phase=mode, tar_attr=tar_attr,
            priv_attr=priv_attr, clr_ratio=clr_ratio, 
            train_path=train_path, test_path=test_path)
        return self._instance

if __name__ == '__main__':
    mode = 'train'
    tar_attr = 'income'
    priv_attr = 'sex'
    train_path = '/home/dnamrata/scratch/data/adult/adult.data'
    test_path = '/home/dnamrata/scratch/data/adult/adult.test'
    dataset = AdultClsDataset(mode, tar_attr, priv_attr, train_path=train_path, test_path=test_path)
    import pdb; pdb.set_trace()
