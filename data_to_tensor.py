import os
import csv
import torch
import cv2
from pydicom import dcmread
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

# Parametros de transformacion de imagen
width, height = 1024,1024
padding = 0.1
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

def get_size(image_file, flip_flag):
    imagen = dcmread(os.path.join(image_dir, image_file)).pixel_array
    if flip_flag:
        imagen = cv2.flip(imagen,1)

    _,img_bin = cv2.threshold(imagen, 0.0,255.0,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,hierarchy = cv2.findContours(img_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(cnts[len(cnts)-1])

    return x,y,w,h

def process_images(image_file, padding, x,y,ancho,alto,flip_flag):
    img_path = os.path.join(image_dir, image_file)
    image = dcmread(img_path).pixel_array

    if flip_flag:
        image = cv2.flip(image,1)
        roi_img = image[y:y+alto+1, x:x+ancho+1]
        roi_img = cv2.flip(roi_img,1)
    else:
        roi_img = image[y:y+alto+1, x:x+ancho+1]
 
    #Parametros para realizar el padding
    # borderType = cv2.BORDER_CONSTANT
    # top = int(padding * roi_img.shape[0])  # imagen[0] = rows
    # bottom = top
    # left = int(padding * roi_img.shape[1])  # imagen[1] = cols
    # right = left
    # roi_img = cv2.copyMakeBorder(roi_img, top, bottom, left, right, borderType, None, value=0.0)
              
    roi_img = cv2.normalize(roi_img, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX).astype(np.float32)
    
    return roi_img

def process_mass_masks(mask_file, padding,x,y,ancho,alto,flip_flag):
    mask_path = os.path.join(mass_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if flip_flag:
        mask = cv2.flip(mask,1)
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]
        roi_mask = cv2.flip(roi_mask,1)
    else:
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]

    #Parametros para realizar el padding
    # borderType = cv2.BORDER_CONSTANT
    # top = int(padding * roi_mask.shape[0])  # imagen[0] = rows
    # bottom = top
    # left = int(padding * roi_mask.shape[1])  # imagen[1] = cols
    # right = left
    # roi_mask = cv2.copyMakeBorder(roi_mask, top, bottom, left, right, borderType, None, value=0.0)
    
    return roi_mask

def process_muscle_masks(mask_file, padding,x,y,ancho,alto,flip_flag):
    mask_path = os.path.join(pec_muscle_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    if flip_flag:
        mask = cv2.flip(mask,1)
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]
        roi_mask = cv2.flip(roi_mask,1)
    else:
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]
    
    #Parametros para realizar el padding
    # borderType = cv2.BORDER_CONSTANT
    # top = int(padding * roi_mask.shape[0])  # imagen[0] = rows
    # bottom = top
    # left = int(padding * roi_mask.shape[1])  # imagen[1] = cols
    # right = left
    # roi_mask = cv2.copyMakeBorder(roi_mask, top, bottom, left, right, borderType, None, value=0.0)

    return roi_mask

def process_metadata(metadata_dir, masa):
    for index in range(len(masa)):
        filename = str(masa).strip('_mask.jpg')

        patient_list = pd.read_csv(metadata_dir)

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
    flip_flag = False
    if os.path.basename(dicom).split('_')[3] == 'R':
        flip_flag = True

    x,y,ancho,alto = get_size(dicom,flip_flag)
    image = process_images(dicom, padding,x,y,ancho,alto,flip_flag)
    mass_mask = process_mass_masks(masas, padding,x,y,ancho,alto,flip_flag)
    muscle_mask = process_muscle_masks(musculos, padding,x,y,ancho,alto,flip_flag)
    metadata = process_metadata(metadata_dir,masas)
    bbox = masks_to_yolo(mass_mask, muscle_mask)

    image = save_to_tensor(image)
    mass_mask = save_to_tensor(mass_mask)
    muscle_mask = save_to_tensor(muscle_mask)

    sample = {'ID':metadata[0], 
              'image': image, 
              'mass_mask': mass_mask,
              'pectoral_muscle_mask':muscle_mask, 
              'Bi-Rads': metadata[1], 
              'ACR': metadata[2], 
              'Side': metadata[3], 
              'View': metadata[4], 
              'bbox':bbox}
    
    tensor_path = os.path.join(end_folder, os.path.basename(os.path.join(image_dir, dicom)).strip('.dcm')+'.pt')
    torch.save(sample, tensor_path, pickle_protocol=pickle.DEFAULT_PROTOCOL)

print('Proceso Finalizado')