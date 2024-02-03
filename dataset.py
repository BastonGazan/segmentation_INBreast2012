import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

class INBreastDataset2012(Dataset):
    def __init__(self, dict_dir, transform=False, target_transform=False):
        self.dict_dir = dict_dir
        self.data = os.listdir(self.dict_dir)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dict_path = os.path.join(self.dict_dir, self.data[index])
        image = torch.load(dict_path)['image']
        mass_mask = torch.load(dict_path)['mass_mask']
        pectoral_muscle_mask = torch.load(dict_path)['pectoral_muscle_mask']


        if self.transform:
            image = self.transform(image)
            pectoral_muscle_mask = self.transform(pectoral_muscle_mask)

        if self.target_transform:
            mass_mask = self.transform(mass_mask)
        
        sample = {'image': image, 'mass_mask':mass_mask, 'pectoral_muscle_mask':pectoral_muscle_mask}

            
        
        return sample
