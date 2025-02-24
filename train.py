import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.transforms import ToTensor
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2

import pdb


dir_img = Path('./data/oi/')
dir_mask = Path('./data/oi_mask/')
dir_checkpoint = Path('./checkpoints23/')

transform = A.Compose([

    #A.Rotate(1),
    #A.RandomScale(0.2)
    #A.Affine(translate_px = (-20,20))
    #A.Affine(shear = (-3,3))
    # ToTensorV2()


])

# transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.5),
#                                 transforms.RandomRotation((-5, 5)),
#                                 transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1)),
#                                 transforms.ColorJitter(brightness=0, contrast=0.4, saturation=0, hue=0)
                                
#                                 ])


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', name = 'test')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    train_loss = []
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                ### 여기 내가 억지로 맞춘 부분
                # images = images[0][0][:][:].unsqueeze(0).unsqueeze(1)
                # print(images.shape)
                # ###

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                
                x1 = images.cpu().detach().numpy()
                

                # #img debugging
                # img = np.zeros((256,256,3))
                # for i in range(1):
                #     for ch in range(3):
                #         img[:, :,ch] = x1[i]
                #     cv2.imwrite('./debug_{}.png'.format(i),img*255)

                true_masks = true_masks.to(device=device, dtype=torch.long)/255



                #pdb.set_trace()
                
                ###여기도 내가 억지로 맞춘 부분
                # true_masks = true_masks[:][:][:][0]
                # true_masks = true_masks.unsqueeze(0)
                # ###
                #print('1', true_masks.shape)

                x2 = true_masks.unsqueeze(0).cpu().detach().numpy()
                

                #image debugging
                # img2 = np.zeros((256,256,3))
                # for i2 in range(1):
                #     for ch2 in range(3):
                #         img2[:, :,ch2] = x2[i2]
                #     cv2.imwrite('./debug2_{}.png'.format(i2),img2*255)
                #print(true_masks.dim())
                #true_masks = true_masks.to(device=device, dtype=torch.long) #torch.long은 integer

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    #print(masks_pred.max())
                    # print(masks_pred)
                    # print(masks_pred.max())
                    masks_pred = masks_pred.squeeze(1)
                    #true_masks = true_masks.unsqueeze(1)
                    #print(true_masks.max())
                    #loss = criterion(masks_pred,true_masks)

                    #print('2',true_masks.shape)
                    #loss = criterion(masks_pred, true_masks)
                    loss = criterion(masks_pred, true_masks) \
                        + dice_loss(masks_pred.float(), true_masks.float())
                
                    # loss = criterion(masks_pred, true_masks) \
                    #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)

                # print(true_masks[0].shape)
                # print(masks_pred[0].shape)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, val_loss = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {} / Validation Loss : {}'.format(val_score, val_loss))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'validation Loss' : val_loss,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred[0].float().cpu()),
                                #'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    #parser.add_argument('-u', '--unet_type', dest='unet_type', metavar='U', type=str, default='v3', help='UNet type is v1/v2/v3 (unet unet++ unet3+)')

    return parser.parse_args()


if __name__ == '__main__':
    wandb.init()
    args = get_args()
    #unet_type = args.unet_type

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #net = UNet(n_channels=1, n_classes=1, bilinear=True) #output channel 2개

    # if unet_type == 'v2':
    #     net = UNet2Plus(n_channels=1, n_classes=1)
    # elif unet_type == 'v3':
    #     net = UNet3Plus(n_channels=1, n_classes=1)
    #     #net = UNet3Plus_DeepSup(n_channels=3, n_classes=1)
    #     #net = UNet3Plus_DeepSup_CGM(n_channels=3, n_classes=1)
    # else:
    #     net = UNet(n_channels=1, n_classes=1)

    net = UNet(n_channels=1, n_classes=1)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
