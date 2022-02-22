""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn
import torch
import numpy as np
import cv2


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x): #[1,1,256,256]
        x1 = self.inc(x) #[1,64,256,256]
        # #train feature extraction check
        # out2 = x1.cpu().detach().numpy()
        # img = np.zeros((256,256,3))
        # for ch in range(3):
        #     img[:, :,ch] = out2[0,0]
        # img = (img/img.max()) * 255
        # img1 = np.array(img, dtype = 'uint8')
        # cv2.imwrite(f'./debug.png',img1)

        x2 = self.down1(x1) #[1,128, 128, 128]

        # #train feature extraction check
        # out2 = x2.cpu().detach().numpy()
        # img = np.zeros((128,128,3))
        # for ch in range(3):
        #     img[:, :,ch] = out2[0,0]
        # img = (img/img.max()) * 255
        # img1 = np.array(img, dtype = 'uint8')
        # cv2.imwrite(f'./debug2.png',img1)

        x3 = self.down2(x2) #[1,,, 64, 64]

        # #train feature extraction check
        # out2 = x3.cpu().detach().numpy()
        # img = np.zeros((64,64,3))
        # for ch in range(3):
        #     img[:, :,ch] = out2[0,0]
        # img = (img/img.max()) * 255
        # img1 = np.array(img, dtype = 'uint8')
        # cv2.imwrite(f'./debug3.png',img1)

        x4 = self.down3(x3) #[1,512, 32, 32]
        
        # #train feature extraction check
        # out2 = x4.cpu().detach().numpy()
        # img = np.zeros((32,32,3))
        # for ch in range(3):
        #     img[:, :,ch] = out2[0,0]
        # img = (img/img.max()) * 255
        # img1 = np.array(img, dtype = 'uint8')
        # cv2.imwrite(f'./debug4.png',img1)

        x5 = self.down4(x4) #[1,512,16,16]

        # out2 = x5.cpu().detach().numpy()
        # img = np.zeros((16,16,3))
        # for ch in range(3):
        #     img[:, :,ch] = out2[0,0]
        # img = (img/img.max()) * 255
        # img1 = np.array(img, dtype = 'uint8')
        # cv2.imwrite(f'./debug5.png',img1)      

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #return logits
        return torch.sigmoid(logits)
