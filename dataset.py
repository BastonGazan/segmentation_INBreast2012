import torch
from torch.utils.data import Dataset
import os

class INBreastDataset2012(Dataset):
    def __init__(self, dict_dir, transform=None):
        self.dict_dir = dict_dir
        self.data = os.listdir(self.dict_dir)
        self.transform = transform



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dict_path = os.path.join(self.dict_dir, self.data[index])
        image = torch.load(dict_path)['image']#.unsqueeze(0)
        mass_mask = torch.load(dict_path)['mass_mask']#.unsqueeze(0)
        pectoral_muscle_mask = torch.load(dict_path)['pectoral_muscle_mask']#.unsqueeze(0)


        if self.transform:
            image, mass_mask, pectoral_muscle_mask = self.transform(image, mass_mask, pectoral_muscle_mask)
            
        
        return image, mass_mask
