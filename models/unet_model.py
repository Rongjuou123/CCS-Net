import torch.nn.functional as F
import torch
from torch import nn

from models.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        # 下采样路径 (8层下采样)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # self.down5 = Down(1024, 1024)  # 新增
        # self.down6 = Down(1024, 1024)  # 新增

        
        # factor = 2 if bilinear else 1
        
        # 上采样路径 - 每个Up模块第一个参数精确对应拼接后的总通道数
        # self.up1 = Up(1024 + 1024, 1024)         # 拼接x9(1024) + x8(1024)
        # self.up2 = Up(1024 + 1024, 1024)         # 拼接up1输出(1024) + x7(1024)
        # self.up3 = Up(1024 + 512, 512)           # 拼接up4输出(1024) + x4(512)
        # self.up4 = Up(512 + 256, 256)            # 拼接up5输出(512) + x3(256)
        # self.up5 = Up(256 + 128, 128)            # 拼接up6输出(256) + x2(128)
        # self.up6 = Up(128 + 64, 64)              # 拼接up7输出(128) + x1(64)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.sig = nn.Sigmoid()
        # self.outf = OutConv(n_channels, n_channels)


        # self.down23_8 = Down(128,256)
        # self.down23_6 = Down(128,128)
        # self.down34 = Down(256,256)
        # self.up43 = Up(512,128, bilinear)
        # self.up32 = Up(256,64, bilinear)
        # self.up21 = Up(128,64, bilinear)
        
    # # ori
    # def forward(self, x):
    #     x1 = self.inc(x)        # 64
    #     x2 = self.down1(x1)     # 128
    #     x3 = self.down2(x2)     # 256
    #     x4 = self.down3(x3)     # 512
    #     x5 = self.down4(x4)     # 1024
    #     x6 = self.down5(x5)     # 1024 (新增)
    #     x7 = self.down6(x6)     # 1024 (新增)
        
    #     # 解码器路径 (上采样)
    #     x = self.up1(x7, x6)    # 1024
    #     x = self.up2(x, x5)     # 1024 (新增)
    #     x = self.up3(x, x4)     # 1024 (新增)
    #     x = self.up4(x, x3)     # 1024 (新增)
    #     x = self.up5(x, x2)     # 512
    #     x = self.up6(x, x1)     # 256

    #     logits = self.outc(x)
    #     # logits = self.sig(logits)
    #     # logits = self.outf(logits)
        
    #     return logits

    def forward(self, x):
        x1 = self.inc(x)        # 64
        x2 = self.down1(x1)     # 128
        x3 = self.down2(x2)     # 256
        x4 = self.down3(x3)     # 512
        x5 = self.down4(x4)     # 1024

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

    # 6 layers
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down23_6(x2)
    #     x = self.up32(x3,x2)
    #     x = self.up21(x, x1)
    #     logits = self.outc(x)
    #     return logits
    
    # # 8 layers
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down23_8(x2)
    #     x4 = self.down34(x3)
    #     x = self.up43(x4,x3)
    #     x = self.up32(x,x2)
    #     x = self.up21(x, x1)
    #     logits = self.outc(x)
    #     return logits
    