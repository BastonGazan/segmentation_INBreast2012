import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import INBreastDataset
from tqdm import tqdm
import pickle

image_dir = r'In_Breast_2012\AllDICOMs'
mask_dir = r'In_Breast_2012\Masks'
metadata_dir = r'In_Breast_2012\INbreast.csv'
end_folder = 'Tensors'

width, height = 240,240

transformed_dataset = INBreastDataset(image_dir=image_dir,
                                    mask_dir=mask_dir,
                                    metadata_dir = metadata_dir,
                                    transform=A.Compose ([
                                            A.Resize(width, height),
                                            ToTensorV2()]))

if not os.path.exists(end_folder):
    os.makedirs(end_folder)

def exportar_tensor(dataset, dicom_path):
    tensor_path = os.path.join(end_folder, os.path.basename(dicom_path).strip('.dcm')+'.pt')
    for i, sample in enumerate(dataset):
        #Exportacion de la imagen a tensor y guardado en la carpeta final
        if sample['ID'] == os.path.basename(dicom_path).split('_')[0]:
            torch.save(sample, tensor_path, pickle_protocol=pickle.DEFAULT_PROTOCOL)
            break

for root,dirs,files in os.walk(image_dir):
    for file in tqdm(files):
        if file.lower().endswith(".dcm"):
            dicom_path  = os.path.join(root,file)
            exportar_tensor(transformed_dataset, dicom_path)

print('Proceso Finalizado')
