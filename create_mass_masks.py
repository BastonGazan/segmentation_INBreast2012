import os
import cv2 as cv
import numpy as np
import plistlib
import skimage.draw
from pydicom import dcmread
from tqdm import tqdm

xml_origin_folder = r'In_Breast_2012\AllXML'
end_folder = r'In_Breast_2012\Mass_Masks'
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
    img = dcmread(os.path.join(dicom_folder,dcm_file)).pixel_array
    imshape = img.shape

    mascara = np.zeros(imshape)

    if os.path.exists(xml_path):
        with open (xml_path,'rb') as mask_file:
            
            i =  0
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
    else:
        mascara = mascara

    return mascara

def export_mask(dcm_file):
    mask = generar_mascara(os.path.join(xml_origin_folder,str(dcm_file).split('_')[0]+'.xml'),dcm_file)

    mask_path = os.path.join(end_folder, str(dcm_file).split('_')[0]+'_mask'+'.jpg')
    cv.imwrite(mask_path, mask)

patients = os.listdir(dicom_folder)

for patient in tqdm(patients):
    print(patient)
    export_mask(patient)

print('Proceso finalizado')