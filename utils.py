import torch
import torchvision
from dataset import INBreastDataset2012
from torch.utils.data import DataLoader, random_split
import os

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('-> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('-> loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(
        train_dir,
        batch_size,
        train_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = INBreastDataset2012(
        train_dir,
        transform=train_transform,
    )
    generator = torch.Generator()
    train_ds, val_ds = random_split(train_ds,lengths=[0.7, 0.3] , generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.unsqueeze(0).float().to(device)
            y = y.unsqueeze(0).float().to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2* (preds * y).sum()) / ((preds + y).sum() + 1e-8)
        
    print(
        f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}'
    )
    print(
        f'Dice score: {dice_score/len(loader)}'
    )
    model.train()

def save_predictions_as_imgs(loader, model, folder ='predicciones_UNET', device ='cuda'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.unsqueeze(0).float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{folder}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(0).float(), f'{folder}/target_{idx}.png')

    model.train()
