import os
import csv
import torch
import cv2
from pydicom import dcmread
from tqdm import tqdm
import pickle
import numpy as np

# Parametros de transformacion de imagen
width, height = 1024,1024
padding = 0.05
interpolation_method = cv2.INTER_AREA

### ================================================== ###

image_dir = r'In_Breast_2012\AllDICOMs'
mass_dir = r'In_Breast_2012\Mass_Masks'
pec_muscle_dir = r'In_Breast_2012\Pectoral_Muscle_Masks'
metadata_dir = r'In_Breast_2012\INbreast.csv'
end_folder = 'Tensors'

images = os.listdir(image_dir)
masa = os.listdir(mass_dir)
pec_muscle = os.listdir(pec_muscle_dir)

def process_images(image_file, padding):
    img_path = os.path.join(image_dir, image_file)
    image = dcmread(img_path).pixel_array

    #Parametros para realizar el padding
    borderType = cv2.BORDER_CONSTANT
    top = int(padding * image.shape[0])  # imagen[0] = rows
    bottom = top
    left = int(padding * image.shape[1])  # imagen[1] = cols
    right = left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType, None, value=0.0)

    image = cv2.normalize(image, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX).astype(np.float32)
    
    return image

def process_mass_masks(mask_file, padding):
    mask_path = os.path.join(mass_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    #Parametros para realizar el padding
    borderType = cv2.BORDER_CONSTANT
    top = int(padding * image.shape[0])  # imagen[0] = rows
    bottom = top
    left = int(padding * image.shape[1])  # imagen[1] = cols
    right = left
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, borderType, None, value=0.0)
    
    return mask

def process_muscle_masks(mask_file, padding):
    mask_path = os.path.join(pec_muscle_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    #Parametros para realizar el padding
    borderType = cv2.BORDER_CONSTANT
    top = int(padding * image.shape[0])  # imagen[0] = rows
    bottom = top
    left = int(padding * image.shape[1])  # imagen[1] = cols
    right = left
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, borderType, None, value=0.0)
    
    return mask

def process_metadata(metadata_dir, masa):
    for index in range(len(masa)):
        filename = str(masa).strip('_mask.jpg')

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
        break

    metadata = [id, bi_rads, acr, side, view]

    return metadata

def masks_to_yolo(masa, musculo):

    bbox = []
    alto, ancho = masa.shape

    cant_mass = len(np.unique(masa))-1
    if cant_mass > 0:
        for i in range(cant_mass):
            copy_mask = masa.copy()

            #Genero una mascara por cada masa que hay en la mascara
            copy_mask[masa!=i+1] = 0
            
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
    
    cant_muscles = len(np.unique(musculo))-1
    if cant_muscles > 0:
        for i in range(cant_muscles):
            copy_mask = musculo.copy()

            #Genero una mascara por cada masa que hay en la mascara
            copy_mask[musculo!=i+1] = 0
            
            #Obtengo las coordenadas que forman el contorno de la masa
            cnts,_ = cv2.findContours(copy_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #Obtengo el rectangulo que abarca el contorno obtenido
            x,y,w,h = cv2.boundingRect(cnts[0])
            
            #Transformo los valores obtenidos del bbox al formato YOLO
            x_centro = x+w//2
            y_centro = y+h//2
            bbox.append([x_centro/ancho, y_centro/alto, w/ancho, h/alto, 'muscle'])
    else:
        bbox = bbox
        
    return bbox

def save_to_tensor(np_array):
    np_array = cv2.resize(np_array,(width,height), interpolation=interpolation_method)
    np_array = torch.from_numpy(np_array)

    return np_array

### =================================================== ###

if not os.path.exists(end_folder):
    os.makedirs(end_folder)

for dicom, masas, musculos in tqdm(zip(images,masa,pec_muscle)):

    image = process_images(dicom, padding)
    mass_mask = process_mass_masks(masas, padding)
    muscle_mask = process_muscle_masks(musculos, padding)
    metadata = process_metadata(metadata_dir,masas)
    bbox = masks_to_yolo(mass_mask, muscle_mask)

    image = save_to_tensor(image)
    mass_mask = save_to_tensor(mass_mask)
    muscle_mask = save_to_tensor(muscle_mask)

    sample = {'ID':metadata[0], 
              'image': image, 
              'mask_mass': mass_mask,
              'pectoral_muscle_mask':muscle_mask, 
              'Bi-Rads': metadata[1], 
              'ACR': metadata[2], 
              'Side': metadata[3], 
              'View': metadata[4], 
              'bbox':bbox}
    
    tensor_path = os.path.join(end_folder, os.path.basename(os.path.join(image_dir, dicom)).strip('.dcm')+'.pt')
    torch.save(sample, tensor_path, pickle_protocol=pickle.DEFAULT_PROTOCOL)

print('Proceso Finalizado')