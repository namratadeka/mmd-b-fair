import torch
import pandas as pd
from PIL import Image
from os.path import join

from torch.utils.data import Dataset
from torchvision import transforms as T


class Waterbirds(Dataset):
    def __init__(self, loc:str, mode:str, factor:str, factor_value:int, 
                 c_factor:str=None, c_factor_value:int=None):
        labels = pd.read_csv(join(loc, 'metadata.csv'))
        if mode == 'train':
            labels = labels[labels['split'] == 0]
        elif mode == 'val':
            labels = labels[labels['split'] == 1]
        elif mode == 'test':
            labels = labels[labels['split'] == 2]
        else:
            raise Exception("mode must be in ['train', 'val', 'test'].")

        self.loc = loc
        labels = labels[labels[factor] == factor_value]

        if c_factor is not None:
            labels = labels[labels[c_factor] == c_factor_value]
        
        self.Y = labels['y'].to_numpy()
        self.S = labels['place'].to_numpy()
        self.images = labels['img_filename'].to_numpy()

        target_resolution = (224, 224)
        if mode == 'train':
            transforms = [
                T.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        else:
            scale = 256.0/224.0
            transforms = [
                T.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                T.CenterCrop(target_resolution),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        self.transform = T.Compose(transforms)

    def __len__(self):
        return self.images.shape[0]
    
    def read_image(self, index):
        img = Image.open(join(self.loc,self.images[index])).convert('RGB')
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        img = self.read_image(index)
        return {
            'data': img,
            'target': torch.tensor(self.Y[index]).long(),
            'sensitive': torch.tensor(self.S[index]).long()
        }

class WaterbirdsBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, loc:str, mode:str, factor:str, factor_value:int, 
                 c_factor:str=None, c_factor_value:int=None, **_ignored):

        self._instance = Waterbirds(loc=loc, mode=mode, factor=factor, factor_value=factor_value,
                                    c_factor=c_factor, c_factor_value=c_factor_value)
        return self._instance