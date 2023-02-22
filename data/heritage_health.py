import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

threshold = {
    'CharlsonIndex_avg': 0,
    'AgeAtFirstClaim': 7
}

class HeritageHealthCI:
    def __init__(self, loc:str, train_path:str, target:str, target_val:list, sensitive:str, sensitive_val:list, ratio:list=None):
        self.target = target
        self.target_vals = target_val
        self.sensitive = sensitive
        self.sensitive_vals = sensitive_val
        self.ratio = ratio
        df = pd.read_csv(loc, header=0)
        
        if target in threshold:
            df[target] = np.where(df[target] > threshold[target], 1, 0)
        df[sensitive] = np.where(df[sensitive] > threshold[sensitive], 1, 0)

        if ratio is not None:
            self._create_data(df)
        else:
            self.df = df
            self.Y = df[target].to_numpy()
            self.A = df[sensitive].to_numpy()
        
        # drop memberID and target label    
        cols2drop = ['Year', 'MemberID', target]
        for col in self.df.columns.tolist():
            if 'prim_' in col and col != target:
                cols2drop.append(col)
        self.df.drop(columns=cols2drop, inplace=True)
        
        self.scaler = MinMaxScaler()
        if 'train' not in loc:
            scale_df = pd.read_csv(train_path, header=0)
            scale_df.drop(columns=cols2drop, inplace=True)
            self.scaler.fit(scale_df)
            del scale_df
        else:
            self.scaler.fit(self.df)

        self.df[self.df.columns] = self.scaler.transform(self.df)
        
    def _create_data(self, df):
        s_labels = df[self.sensitive].to_numpy()
        t_labels = df[self.target].to_numpy()

        valid_s0 = np.where(s_labels == self.sensitive_vals[0])[0]
        valid_s1 = np.where(s_labels == self.sensitive_vals[1])[0]

        valid_t0 = np.where(t_labels == self.target_vals[0])[0]
        valid_t1 = np.where(t_labels == self.target_vals[1])[0]

        N = df.shape[0]
        up_ineligible_idx = np.array(list(set(valid_t0).intersection(set(valid_s0))))
        p_ineligible_idx = np.array(list(set(valid_t0).intersection(set(valid_s1))))

        up_eligible_idx = np.array(list(set(valid_t1).intersection(set(valid_s0))))
        p_eligible_idx = np.array(list(set(valid_t1).intersection(set(valid_s1))))

        target_ratio = valid_t1.shape[0] / (valid_t1.shape[0] + valid_t0.shape[0])

        indices = -1 * np.ones(N, dtype=int)
        classes = np.random.binomial(1, target_ratio, N)
        for i in range(len(indices)):
            if classes[i]:
                if np.random.random() <= self.ratio[1]:
                    indices[i] = np.random.choice(up_eligible_idx)
                else:
                    indices[i] = np.random.choice(p_eligible_idx)
            else:
                if np.random.random() <= self.ratio[0]:
                    indices[i] = np.random.choice(up_ineligible_idx)
                else:
                    indices[i] = np.random.choice(p_ineligible_idx)

        self.df = df.iloc[indices]
        self.Y = self.df[self.target].to_numpy()
        self.A = self.df[self.sensitive].to_numpy()

class HeritageHealthDataset(Dataset):
    def __init__(self, loc, train_path, factor, factor_value, target, 
                target_val, sensitive, sensitive_val, c_factor:str=None, 
                c_factor_value:str=None ,ratio=None) -> None:
        data = HeritageHealthCI(loc, train_path, target, target_val, sensitive, sensitive_val, ratio)

        if factor == 'AgeAtFirstClaim':
            att = 'A'
        if factor == 'CharlsonIndex_avg':
            att = 'Y'
        indices = np.where(eval('data.{}'.format(att)) == factor_value)[0]
        self.data = data.df.iloc[indices]
        self.Y = data.Y[indices]
        self.A = data.A[indices]

        if c_factor is None:
            c_factor = factor
            c_factor_value = factor_value
        if c_factor == 'AgeAtFirstClaim':
            c_att = 'A'
        if c_factor == 'CharlsonIndex_avg':
            c_att = 'Y'
        mask = eval(f'self.{c_att}') == c_factor_value
        self.data = self.data.iloc[mask]
        self.A = self.A[mask]
        self.Y = self.Y[mask]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {
            'data': torch.from_numpy(self.data.iloc[index].to_numpy()).float(),
            'target': torch.tensor(self.Y[index]).long(),
            'sensitive': torch.tensor(self.A[index]).long()
        }

class HeritageHealthDatasetBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, loc:str, train_path:str, factor:str, factor_value:int, target:str, 
                target_val:list, sensitive:str, sensitive_val:list, 
                c_factor:str=None, c_factor_value:str=None, ratio:list=None, 
                **_ignored):

        self._instance = HeritageHealthDataset(
            loc=loc, train_path=train_path, factor=factor, factor_value=factor_value, 
            target=target, target_val=target_val, sensitive=sensitive, 
            sensitive_val=sensitive_val, c_factor=c_factor, 
            c_factor_value=c_factor_value, ratio=ratio)
        return self._instance
