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
        patient_dict = torch.load(dict_path)
        image = patient_dict['image'].unsqueeze(0)
        mass_mask = patient_dict['mass_mask'].unsqueeze(0)


        if self.transform:
            image, mass_mask = self.transform(image, mass_mask)
            
        
        return image, mass_mask
