import torch
import pandas as pd

from torch.utils.data import Dataset


class CompasCls(Dataset):
    def __init__(self, path:str, target:str, sensitive:str):
        self.data = pd.read_csv(path)
        self.A = self.data[sensitive].to_numpy()
        self.Y = self.data[target].to_numpy()

        self.data.drop(columns=[target], inplace=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data.iloc[index].to_numpy()).float(),
            torch.tensor(self.Y[index]).long(),
            torch.tensor(self.A[index]).long()
        )

class CompasClsBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, path:str, target:str, sensitive:str, **_ignored):

        self._instance = CompasCls(path=path, target=target, sensitive=sensitive)
        return self._instance
