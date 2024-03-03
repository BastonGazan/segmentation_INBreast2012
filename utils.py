import torch
import torchvision
from dataset import INBreastDataset2012
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

def calc_dice_coeficient(pred, target):
  dice_score = 0.
  intersection = torch.sum(pred * target)
  union = torch.sum(pred) + torch.sum(target)
  if union == 0.0:
    dice_score = 1.0
  else:
    dice_score = torch.mean((2. * intersection) / union)

  return dice_score

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
    generator = torch.Generator().manual_seed(42)
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
        shuffle=True,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, folder ='predicciones_UNET/', device="cuda"):
    print('-> Checking accuracy')
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader)):
            x = x.unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += calc_dice_coeficient(preds, y)
            print('-> Saving predictions as images')
            torchvision.utils.save_image(preds, f'{folder}/pred_{idx}.png')
            torchvision.utils.save_image(y.float(), f'{folder}/target_{idx}.png')

    print(
        f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}'
    )
    print(
        f'Dice score: {dice_score/len(loader)}'
    )
    model.train()
