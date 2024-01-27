import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from pydicom import dcmread
import torch

class INBreastDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return(len(self.images))
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = cv2.normalize(dcmread(img_path).pixel_array, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        sample = {'image': image, 'mask': mask}

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            sample = {'image': image, 'mask': mask}

        
        return sample


    