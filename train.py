import torch
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode
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
    save_predictions_as_imgs,
)

# Hyperparametros de entrenamiento
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 3
NUM_WORKERS = 1
HEIGHT = 240
WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = r'Tensors\Imagenes_centradas\train'
TEST_DIR = r'Tensors\Imagenes_centradas\test'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.unsqueeze(0).float().to(device=DEVICE)
        targets = targets.unsqueeze(0).float().to(device=DEVICE)

        predictions = model(data)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = T.Compose(
        [
            T.RandomRotation(degrees=35, expand=True, fill=0.0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.Resize((HEIGHT,WIDTH), interpolation=InterpolationMode.NEAREST_EXACT)
        ]
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # Investigar
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_DIR,
        BATCH_SIZE,
        train_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)
        check_accuracy(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader,model,optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(val_loader, model, folder=r"predicciones_UNET/",device=DEVICE)

        

if __name__ == '__main__':
    main()

