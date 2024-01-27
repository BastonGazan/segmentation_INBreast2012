import os
import cv2
import numpy as np
from pydicom import dcmread
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from dataset import INBreastDataset
from tqdm import tqdm

image_dir = r'In_Breast_2012\AllDICOMs\train_images'
mask_dir = r'In_Breast_2012\massMasks\train_masks'
end_folder = 'train_tensors'

width, height = 240,240

transformed_dataset = INBreastDataset(image_dir=image_dir,
                                mask_dir=mask_dir,
                                transform=A.Compose ([
                                    A.Resize(width, height),
                                    ToTensorV2()]))


if not os.path.exists(end_folder):
    os.makedirs(end_folder)

def exportar_tensor(dataset, dicom_path):
    for i, sample in enumerate(dataset):

    #Exportacion de la imagen a jpg y guardado en la carpeta final
        tensor_path = os.path.join(end_folder, os.path.basename(dicom_path).strip('.dcm')+'.pt')
        torch.save(sample, tensor_path)

for root,dirs,files in os.walk(image_dir):
    for file in tqdm(files):
        if file.lower().endswith(".dcm"):
            dicom_path  = os.path.join(root,file)
            exportar_tensor(transformed_dataset, dicom_path)

print('Proceso Finalizado')
