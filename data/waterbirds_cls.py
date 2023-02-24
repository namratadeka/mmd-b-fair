import torch
import pandas as pd
from PIL import Image
from os.path import join

from torch.utils.data import Dataset
from torchvision import transforms as T


class WaterbirdsCls(Dataset):
    def __init__(self, path:str, mode:str):
        labels = pd.read_csv(join(path, 'metadata.csv'))
        if mode == 'train':
            labels = labels[labels['split'] == 0]
        elif mode == 'val':
            labels = labels[labels['split'] == 1]
        elif mode == 'test':
            labels = labels[labels['split'] == 2]
        else:
            raise Exception("mode must be in ['train', 'val', 'test'].")

        self.path = path
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
        return len(self.images)

    def read_image(self, index):
        img = Image.open(join(self.path,self.images[index])).convert('RGB')
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        img = self.read_image(index)
        return (
            img,
            torch.tensor(self.Y[index]).long(),
            torch.tensor(self.S[index]).long()
        )

class WaterbirdsClsBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, path:str, mode:str, **_ignored):

        self._instance = WaterbirdsCls(path=path, mode=mode)
        return self._instance