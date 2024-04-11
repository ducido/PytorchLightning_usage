from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset
import cv2
import pandas as pd
import glob
import os
from torchvision import transforms
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class PascalVOC(Dataset):
    def __init__(self, data_dir, types, transform= None):
        self.data_dir = data_dir
        self.preprocess(data_dir + f'annotations/{types}.txt')
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def preprocess(self, ann_path):
        self.df = pd.read_csv(ann_path, names= ['image', 'label', 'species', 'breed'], sep= ' ')

    def __getitem__(self, index):
        img_name = self.df.iloc[index].iloc[0]
        label = self.df.iloc[index].iloc[1] - 1

        img_path = self.data_dir + 'images/' + img_name + '.jpg'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        # label = torch.tensor(label, dtype= torch.long)
        return img, label
    
class Pascal_VOC_LN(pl.LightningDataModule):
    def __init__(self, data_dir, test_data_dir,batch_size, num_workers= 1):
        super().__init__()
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform =  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224,224)),
                ])

    def setup(self, stage):
        self.train_dataset = PascalVOC(data_dir= self.data_dir, types= 'trainval', transform= self.transform)
        self.val_dataset = PascalVOC(data_dir= self.test_data_dir, types= 'test', transform= self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size= self.batch_size, shuffle= True, num_workers= self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size= self.batch_size, shuffle= False, num_workers= self.num_workers)

if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])
    data_dir = '/media/minhduc/D/AAAAA/Pixta/archive/voctrainval_06-nov-2007/VOCdevkit/VOC2007/'
    dataset = PascalVOC(data_dir, types= 'train', transform= transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size= 32, shuffle= True)

    for x, y in train_loader:
        print(x.shape)
        print(y.shape)