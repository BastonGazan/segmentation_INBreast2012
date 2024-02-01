import os
import csv
import torch
import cv2
from pydicom import dcmread
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pickle
import numpy as np

image_dir = r'In_Breast_2012\AllDICOMs'
mask_dir = r'In_Breast_2012\Masks'
metadata_dir = r'In_Breast_2012\INbreast.csv'
end_folder = 'Tensors'
width, height = 1024,1024
padding = 5

images = os.listdir(image_dir)
masks = os.listdir(mask_dir)

def process_images(image_file):

    img_path = os.path.join(image_dir, image_file)
    image = cv2.normalize(dcmread(img_path).pixel_array, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX).astype(np.float32)
    image = torch.from_numpy(image)
    
    return image

def process_masks(mask_file):

    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    mask = torch.from_numpy(mask)
    
    return mask

def process_metadata(metadata_dir):
    for index in range(len(masks)):
        filename = masks[index].strip('_mask.jpg')

        with open(metadata_dir, newline='') as csvfile:
            patient_list = csv.DictReader(csvfile, delimiter=';', quotechar='|')
            for row in patient_list:
                if row['File Name'] == filename:
                    bi_rads = row['Bi-Rads']
                    acr = row['ACR']
                    side = row['Laterality']
                    view = row['View']
                    id = row['File Name']

                    break
    metadata = [id, bi_rads, acr, side, view]

    return metadata

def masks_to_yolo(mask_dir):

    masks = os.listdir(mask_dir)
    for index in range(len(masks)):
        mask_path = os.path.join(mask_dir, masks[index])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

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
        
    return bbox

### =================================================== ###

if not os.path.exists(end_folder):
    os.makedirs(end_folder)

for dicom, mascaras in zip(images,masks):
    image = process_images(dicom)
    mask = process_masks(mascaras)
    metadata = process_metadata(metadata_dir)
    bbox = masks_to_yolo()

    sample = {'ID':metadata[0], 
              'image': image, 
              'mask_mass': mask, 
              'Bi-Rads': metadata[1], 
              'ACR': metadata[2], 
              'Side': metadata[3], 
              'View': metadata[4], 
              'bbox':bbox}

    tensor_path = os.path.join(end_folder, os.path.basename(os.path.join(image_dir, dicom)).strip('.dcm')+'.pt')
    torch.save(sample, tensor_path, pickle_protocol=pickle.DEFAULT_PROTOCOL)

print('Proceso Finalizado')