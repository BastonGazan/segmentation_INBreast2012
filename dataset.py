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
        image = patient_dict['image'].unsqueeze(0).float()
        mass_mask = patient_dict['mass_mask'].unsqueeze(0).float()
        image = image/255.0
        mass_mask[mass_mask < 0.5] = 0.0
        mass_mask[mass_mask >= 0.5] = 1.0


        if self.transform is not None:
            # Concateno las imagenes por la dimension del canal (Hago una imagen de 2 canales)
            image_and_mask = torch.cat([image, mass_mask], dim=0)
            # Puedo pasar una sola imagen y aplicarle la misma transformacion a ambas
            transformed = self.transform(image_and_mask)
            # Separo los tensores
            image = transformed[0,:, :]
            mass_mask = transformed[1,:, :]


        return image, mass_mask
