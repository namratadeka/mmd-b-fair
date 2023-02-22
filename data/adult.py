import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.adult_process import get_adult_data


class AdultDataset:
    """
    The UCI Adult dataset.
    """

    def __init__(self, root_dir, phase, tar_attr, priv_attr, clr_ratio, train_path, test_path):
        self.tar_attr = tar_attr
        self.priv_attr = priv_attr

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

    def onehot_2_int(self, ts):
        if len(ts.shape) == 2:
            return torch.argmax(ts, dim=1)
        if len(ts.shape) == 1:
            return torch.argmax(ts, dim=0)
        raise NotImplementedError

    def get_A_proportions(self):
        """for catergorical attribute"""
        assert len(self.A.shape) == 2
        num_class = self.A.shape[1]

        A_label = np.argmax(self.A, axis=1)
        A_proportions = []
        for cls_idx in range(num_class):
            A_proportion = np.sum(cls_idx == A_label)
            A_proportions.append(A_proportion)
        A_proportions = [a_prop * 1.0 / len(A_label) for a_prop in A_proportions]
        return A_proportions

    def get_Y_proportions(self):
        """for catergorical attribute"""
        assert len(self.Y.shape) == 2
        num_class = self.Y.shape[1]

        Y_label = np.argmax(self.Y, axis=1)
        Y_proportions = []
        for cls_idx in range(num_class):
            Y_proportion = np.sum(cls_idx == Y_label)
            Y_proportions.append(Y_proportion)
        Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
        return Y_proportions

    def get_AY_proportions(self):
        """for catergorical attributes"""
        assert len(self.Y.shape) == len(self.A.shape) == 2
        A_num_class = self.A.shape[1]
        Y_num_class = self.Y.shape[1]
        A_label = np.argmax(self.A, axis=1)
        Y_label = np.argmax(self.Y, axis=1)
        AY_proportions = []
        for A_cls_idx in range(A_num_class):
            Y_proportions = []
            for Y_cls_idx in range(Y_num_class):
                AY_proprtion = np.sum(
                    np.logical_and(Y_cls_idx == Y_label, A_cls_idx == A_label)
                )
                Y_proportions.append(AY_proprtion)
            Y_proportions = [y_prop * 1.0 / len(Y_label) for y_prop in Y_proportions]
            AY_proportions.append(Y_proportions)
        return AY_proportions


class Adult(Dataset):
    '''
    sensitive: age, target: income
    '''
    def __init__(self, sensitive:str, factor:str, factor_value:int, mode:str, clr_ratio:list, train_path:str, test_path:str):
        data = AdultDataset('', mode, 'income', sensitive, clr_ratio, train_path, test_path)
        if factor == 'income':
            att = 'Y'
        else:
            att = 'A'
        indices = np.where(eval('data.{}'.format(att)) == factor_value)[0]
        self.data = data.X[indices]
        self.Y = data.Y[indices]
        self.A = data.A[indices]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {
            'data': torch.from_numpy(self.data[index]).float(),
            'target': torch.tensor(self.Y[index]),
            'sensitive': torch.tensor(self.A[index])
        }

class AdultCond(Dataset):
    '''
    sensitive: age, target: income
    '''
    def __init__(self, sensitive: str, factor:str, factor_value:int, mode:str, clr_ratio:list, train_path:str, test_path:str, c_factor:str, c_factor_value:int):
        data = AdultDataset('', mode, 'income', sensitive, clr_ratio, train_path, test_path)
        if factor == 'income':
            att = 'Y'
        else:
            att = 'A'
        if c_factor == 'income':
            cond_att = 'Y'
        else:
            cond_att = 'A'
        indices = np.where(eval('data.{}'.format(att)) == factor_value)[0]
        tmp_data = data.X[indices]
        tmp_Y = data.Y[indices]
        tmp_A = data.A[indices]

        # condition on c_factor:
        indices = np.where(eval('tmp_{}'.format(cond_att)) == c_factor_value)[0]
        self.data = tmp_data[indices]
        self.Y = tmp_Y[indices]
        self.A = tmp_A[indices]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {
            'data': torch.from_numpy(self.data[index]).float(),
            'target': torch.tensor(self.Y[index]),
            'sensitive': torch.tensor(self.A[index])
        }

class AdultBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, sensitive:str, factor:str, factor_value:int, mode:str,
                 clr_ratio:list, train_path:str, test_path:str, **_ignored):

        self._instance = Adult(sensitive=sensitive,
            factor=factor, factor_value=factor_value, clr_ratio=clr_ratio, mode=mode, 
            train_path=train_path, test_path=test_path)
        return self._instance

class AdultCondBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, sensitive:str, factor:str, factor_value:int, mode:str,
                 clr_ratio:list, train_path:str, test_path:str, 
                 c_factor:str, c_factor_value:int, **_ignored):

        self._instance = AdultCond(sensitive=sensitive,
            factor=factor, factor_value=factor_value, clr_ratio=clr_ratio, 
            mode=mode, train_path=train_path, test_path=test_path, 
            c_factor=c_factor, c_factor_value=c_factor_value)
        return self._instance

if __name__ == '__main__':
    dataset = Adult(sensitive='sex', factor='sex', factor_value=0, mode='val', clr_ratio=None, 
                    train_path='/home/dnamrata/scratch/data/adult/adult.data', 
                    test_path='/home/dnamrata/scratch/data/adult/adult.test')
    import pdb; pdb.set_trace()
