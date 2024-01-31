import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from pydicom import dcmread
import torch
import csv

class INBreastDataset(Dataset):
    def __init__(self, image_dir, mask_dir, metadata_dir,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata_dir = metadata_dir
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
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        #mask[mask > 0.0] = 1.0

        filename = self.masks[index].strip('_mask.jpg')

        with open(self.metadata_dir, newline='') as csvfile:
            patient_list = csv.DictReader(csvfile, delimiter=';', quotechar='|')
            for row in patient_list:
                if row['File Name'] == filename:
                    bi_rads = row['Bi-Rads']
                    acr = row['ACR']
                    side = row['Laterality']
                    view = row['View']
                    id = row['File Name']

                    break
        
        bbox = []
        alto, ancho = mask.shape
        cant_mass = len(np.unique(mask))-1

        if cant_mass > 0:
            for i in range(cant_mass):
                copy_mask = mask.copy()

                #Genero una mascara por cada masa que hay en la mascara
                copy_mask[mask!=i+1] = 0
                
                #Obtengo las coordenadas que forman el contorno de la masa
                cnts,_ = cv2.findContours(copy_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #Obtengo el rectangulo que abarca el contorno obtenido
                x,y,w,h = cv2.boundingRect(cnts[0])
                
                #Transformo los valores obtenidos del bbox al formato YOLO
                x_centro = x+w//2
                y_centro = y+h//2
                bbox.append([x_centro/ancho, y_centro/alto, w/ancho, h/alto, 'mass'])
        else:
            bbox = bbox

        sample = {'ID':id, 'image': image, 'mask': mask, 'Bi-Rads': bi_rads, 'ACR': acr, 'Side': side, 'View': view, 'bbox':bbox}

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return sample
    