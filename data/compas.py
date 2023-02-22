import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class Compas(Dataset):
    def __init__(self, loc:str, factor:str, factor_value, c_factor:str=None, c_factor_value=None):
        df = pd.read_csv(loc)
        Y = df['result'].to_numpy()
        A = df['race'].to_numpy()

        indices = np.where(df[factor].to_numpy() == factor_value)[0]
        Y = Y[indices]
        A = A[indices]
        df = df.iloc[indices]

        if c_factor is not None:
            indices = np.where(df[c_factor].to_numpy() == c_factor_value)[0]
            Y = Y[indices]
            A = A[indices]
            df = df.iloc[indices]

        self.df = df.drop(columns=['result'])
        self.Y = Y
        self.A = A

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return {
            'data': torch.from_numpy(self.df.iloc[index].to_numpy()).float(),
            'target': torch.tensor(self.Y[index]).long(),
            'sensitive': torch.tensor(self.A[index]).long()
        }

class CompasBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, loc:str, factor:str, factor_value:str, c_factor:str=None, c_factor_value:str=None, **_ignored):

        self._instance = Compas(loc=loc, factor=factor, factor_value=factor_value, 
                                c_factor=c_factor, c_factor_value=c_factor_value)
        return self._instance
