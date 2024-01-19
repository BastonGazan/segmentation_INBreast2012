import subprocess

# Lista de librerias para instalar
librerias_para_instalar = [
    'numpy',
    'pandas',
    'matplotlib',
    'pydicom',
    'pillow',
    'pyjpegls',
    'python-gdcm',
    ' -U pylibjpeg[all]',
    'opencv-python',
    'torch torchvision torchaudio',
]

#Instalador
for libreria in librerias_para_instalar:
    try:
        subprocess.check_call(['pip','install',libreria])
        print(f'{libreria} instalada')
    except subprocess.CalledProcessError:
        print(f'Error al instalar {libreria}')