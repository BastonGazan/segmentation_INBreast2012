import torch
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from sklearn.preprocessing import StandardScaler


from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
)

# Hyperparametros de entrenamiento
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 20
NUM_WORKERS = 1
HEIGHT = 1024
WIDTH = 1024
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = r'Tensors\Imagenes_centradas\prueba'
TEST_DIR = r'/content/drive/Shareddrives/PI Bazán-Merino/Entrenamiento/Datos_de_entrenamiento/Tensores_Imagenes_centradas/test'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.unsqueeze(1).to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE)
        optimizer.zero_grad()

#Hasta acá le estoy pasando datos de tipo float32 que van de 0.0 a 1.0
        predictions = model(data)
        folder = 'Tensors\Imagenes_centradas'
        save_image(data,  f'{folder}/data_{batch_idx}.png')
        save_image(predictions, f'{folder}/pred_{batch_idx}.png')
        save_image(targets.float(), f'{folder}/target_{batch_idx}.png')
        print(f'\nmax value prediccion train {predictions.max()}')

        # Calcular Loss y realizar backPropagation
        loss = loss_fn(predictions, targets)
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


        loop.set_postfix(loss=loss.item())

def main():
    train_transform = T.Compose(
        [
            #T.Normalize(mean=[0.0], std= [1.0]),
            T.RandomRotation(degrees=35, expand=True, fill=0.0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.Resize((HEIGHT,WIDTH), interpolation=InterpolationMode.NEAREST_EXACT),
        ]
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss() # Investigar
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_DIR,
        BATCH_SIZE,
        train_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load('/content/drive/Shareddrives/PI Bazán-Merino/Entrenamiento/my_checkpoint.pth.tar'), model)
        check_accuracy(val_loader, model, device=DEVICE)

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = StandardScaler()

    for epoch in range(NUM_EPOCHS):
        print(f'Training epoch {epoch+1} of {NUM_EPOCHS}')
        train_fn(train_loader,model,optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, folder=r"predicciones_UNET", device=DEVICE)

    print(f'Entrenamiento finalizado')


if __name__ == '__main__':
  main()

