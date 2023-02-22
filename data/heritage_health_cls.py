import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

threshold = {
    'CharlsonIndex_avg': 0,
    'AgeAtFirstClaim': 7
}

class HeritageHealthCls(Dataset):
    def __init__(self, path:str, target:str, sensitive:str, og_target:str, train_path:str):
        self.df = pd.read_csv(path, header=0)
        self.Y = self.df[target].to_numpy()
        if target in threshold:
            self.Y = np.where(self.Y > threshold[target], 1, 0)
        self.A = self.df[sensitive].to_numpy()
        self.A = np.where(self.A > threshold[sensitive], 1, 0)
        
        cols2drop = ['Year', 'MemberID', target, og_target]
        for col in self.df.columns.tolist():
            if 'prim_' in col and col != target:
                cols2drop.append(col)
        self.df.drop(columns=cols2drop, inplace=True)

        self.scaler = MinMaxScaler()
        if 'train' not in path:
            scale_df = pd.read_csv(train_path, header=0)
            scale_df.drop(columns=cols2drop, inplace=True)
            self.scaler.fit(scale_df)
            del scale_df
        else:
            self.scaler.fit(self.df)
        
        self.df = self.scaler.transform(self.df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.df[index]).float(),
            torch.tensor(self.Y[index]).long(),
            torch.tensor(self.A[index]).long()
        )

class HeritageHealthClsBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path:str, target:str, sensitive:str, og_target:str, train_path:str, **_ignored):

        self._instance = HeritageHealthCls(path=path, target=target, 
                                sensitive=sensitive, og_target=og_target,
                                train_path=train_path)
        return self._instance
