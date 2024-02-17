import os
import torch
import cv2
from pydicom import dcmread
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

# Parametros de transformacion de imagen
width, height = 1024,1024 # Dimensiones de resizing 
interpolation_method = cv2.INTER_AREA # Metodo de interpolacion para resizing
clipLimit = 5 # Cantidad maxima de pixeles que puede haber de una intensidad

### =====================Directorios============================= ###

image_dir = r'In_Breast_2012\AllDICOMs'
mass_dir = r'In_Breast_2012\Mass_Masks'
pec_muscle_dir = r'In_Breast_2012\Pectoral_Muscle_Masks'
metadata_dir = r'In_Breast_2012\INbreast.csv'
end_folder = 'Tensors'
images = os.listdir(image_dir)
masa = os.listdir(mass_dir)
pec_muscle = os.listdir(pec_muscle_dir)

### ===================Funciones============================== ###

def get_size(image_file, flip_flag):
    imagen = dcmread(os.path.join(image_dir, image_file)).pixel_array
    if flip_flag:
        imagen = cv2.flip(imagen,1)

    _,img_bin = cv2.threshold(imagen, 0.0,255.0,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,hierarchy = cv2.findContours(img_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(cnts[-1])

    return x,y,w,h

def centrar_imagen(imagen):

    borderType = cv2.BORDER_CONSTANT

    alto, ancho = imagen.shape

    if ancho > alto:
        top = int((ancho - alto)//2)
        bottom = top
        left = 0 
        right = left
        imagen = cv2.copyMakeBorder(imagen, top, bottom, left, right, borderType, None, value=0.0)
    elif alto > ancho:
        top = 0
        bottom = top
        left = int((alto - ancho)//2)
        right = left
        imagen = cv2.copyMakeBorder(imagen, top, bottom, left, right, borderType, None, value=0.0)
    else:
        imagen=imagen

    return imagen

def clahe_equalization(image_file,clip):
    clahe = cv2.createCLAHE(clipLimit=clip)
    equ_img = clahe.apply(image_file)

    return equ_img

def process_images(image_file, x,y,ancho,alto,flip_flag,clip):
    img_path = os.path.join(image_dir, image_file)
    image = dcmread(img_path).pixel_array

    if flip_flag:
        image = cv2.flip(image,1)
        roi_img = image[y:y+alto+1, x:x+ancho+1]
        roi_img = cv2.flip(roi_img,1)
    else:
        roi_img = image[y:y+alto+1, x:x+ancho+1]

    roi_img = centrar_imagen(roi_img)

    roi_img = clahe_equalization(roi_img, clip)
              
    roi_img = cv2.normalize(roi_img, None, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    return roi_img

def process_mass_masks(mask_file,x,y,ancho,alto,flip_flag):
    mask_path = os.path.join(mass_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

    if flip_flag:
        mask = cv2.flip(mask,1)
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]
        roi_mask = cv2.flip(roi_mask,1)
    else:
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]

    roi_mask = centrar_imagen(roi_mask)
    
    return roi_mask

def process_muscle_masks(mask_file,x,y,ancho,alto,flip_flag):
    mask_path = os.path.join(pec_muscle_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    
    if flip_flag:
        mask = cv2.flip(mask,1)
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]
        roi_mask = cv2.flip(roi_mask,1)
    else:
        roi_mask = mask[y:y+alto+1, x:x+ancho+1]

    roi_mask = centrar_imagen(roi_mask)
    
    return roi_mask

def process_metadata(metadata_dir, masa):

    filename = int(str(masa).strip('_mask.jpg'))
    patient_list = pd.read_csv(metadata_dir, sep=';')

    paciente = patient_list.loc[patient_list['File Name'] == filename]
    bi_rads = paciente['Bi-Rads'].iloc[0]
    acr = paciente['ACR'].iloc[0]
    side = paciente['Laterality'].iloc[0]
    view = paciente['View'].iloc[0]
    id = paciente['File Name'].iloc[0]

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

### =====================Main============================== ###

if not os.path.exists(end_folder):
    os.makedirs(end_folder)

for dicom, masas, musculos in tqdm(zip(images,masa,pec_muscle)):
    flip_flag = False
    if os.path.basename(dicom).split('_')[3] == 'R':
        flip_flag = True

    x,y,w,h = get_size(dicom,flip_flag)
    image = process_images(dicom,x,y,w,h,flip_flag,clipLimit)
    mass_mask = process_mass_masks(masas,x,y,w,h,flip_flag)
    muscle_mask = process_muscle_masks(musculos,x,y,w,h,flip_flag)
    metadata = process_metadata(metadata_dir,masas)
    bbox = masks_to_yolo(mass_mask, muscle_mask)

    # image = save_to_tensor(image)
    mass_mask = cv2.resize(mass_mask,(width,height), interpolation=interpolation_method)
    muscle_mask = cv2.resize(muscle_mask,(width,height), interpolation=interpolation_method)

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