import os
import cv2 as cv
import numpy as np
import plistlib
import skimage.draw
from pydicom import dcmread
import glob
from tqdm import tqdm

xml_origin_folder = r'In_Breast_2012\AllXML'
end_folder = r'In_Breast_2012\Masks01'
dicom_folder = r'In_Breast_2012\AllDICOMs'

if not os.path.exists(end_folder):
    os.makedirs(end_folder)

def cargar_puntos(point_string):
    x,y = tuple([float(i) for i in point_string.strip('()').split(',')])
    return y,x

def generar_mascara(xml_path,dcm_file):
    """
    Genera una mascara a partir de un archivo XML formato plist de apple
    
    @xml_path : Directorio del archivo xml
    @dcm_file : Archivo dicom donde se extrae el shape que va a tener la mascara
     
    return: numpy array donde cada masa tiene un valor distinto en la mascara
    """
    ds = dcmread(dcm_file)
    img = ds.pixel_array
    imshape = img.shape

    i =  0
    mascara = np.zeros(imshape)
    with open (xml_path,'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML, dict_type=dict)['Images'][0]
        rois = plist_dict['ROIs']
        for roi in rois:
            if roi["Name"] == "Mass":
                i += 1
                # Convierto los elementos de la lista points de <str de coordenadas> a <tuplas de coordenadas>
                points = roi['Point_px']
                points = [cargar_puntos(point) for point in points]
                
                # Genero una mascara con todas las coordenadas del archivo XML
                x, y = zip(*points)
                x, y = np.array(x), np.array(y)
                poly_x, poly_y = skimage.draw.polygon(x,y, shape=imshape)
                mascara[poly_x, poly_y] = i
    return mascara

def crear_lista_mascaras(xml_folder):
    masks_ID = os.listdir(xml_folder)
    for i in range(len(masks_ID)):
        masks_ID[i] = masks_ID[i].strip('.xml')
    return masks_ID

def export_mask(patient_id):
    dcm_path = glob.glob(os.path.join(dicom_folder,str(patient_id)+'*.dcm'))
    mask = generar_mascara(os.path.join(xml_origin_folder,str(patient_id)+'.xml'),dcm_path[0])

    mask_path = os.path.join(end_folder, os.path.basename(patient_id)+'_mask'+'.jpg')
    cv.imwrite(mask_path, mask)

masks_ID = crear_lista_mascaras(xml_origin_folder)

for patient in tqdm(masks_ID):
    export_mask(patient)

print('Proceso finalizado')