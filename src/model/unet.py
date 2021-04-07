# -*- coding: utf-8 -*-

import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from globalenv import *

from .basemodel import BaseModel


class UNetLitModel(BaseModel):
    def __init__(self):
        super().__init__(opt, [TRAIN, VALID])

        self.net = UNetModel()
        self.train_metrics = {
            L1_LOSS: 0,
            # LTV_LOSS: 0,
            # COS_LOSS: 0,
            # LOSS: 0
        }

        self.valid_metrics = {
            L1_LOSS: 0,
            PSNR: 0,
            SSIM: 0
        }

        self.illumination_dirpath = os.path.join(opt[LOG_DIRPATH], PREDICT_ILLUMINATION)
        if opt[RUNTIME][PREDICT_ILLUMINATION]:
            uitl.mkdir(self.illumination_dirpath)

        self.criterion = nn.MSELoss()
        self.net.train()

    def training_step(self, batch, batch_idx):
        input_batch, gt_batch, fpaths = batch[INPUT], batch[GT], batch[FPATH]
        output_batch = self.net(input_batch).clamp(0, 1)
        loss = self.criterion(output_batch, gt_batch)

        # calculate illumination here if needed.

        # logging
        self.train_metrics[L1_LOSS] = loss
        self.log_dict(self.train_metrics)
        self.log_IOG_images(TRAIN, self.global_step, osp.basename(fpaths[0]), input_batch, output_batch, gt_batch)

        return loss

    def validation_step(self):
        # TODO: finish here.
        pass


class UNet(nn.Module):

    def __init__(self):
        """UNet implementation

        :returns: N/A
        :rtype: N/A

        """
        super().__init__()

        self.conv1 = nn.Conv2d(16, 64, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)

        self.local_net = LocalNet(16)

        self.dconv_down1 = LocalNet(3, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 64)
        self.dconv_down4 = LocalNet(64, 128)
        self.dconv_down5 = LocalNet(128, 128)

        self.maxpool = nn.MaxPool2d(2, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_conv1x1_1 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_2 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_3 = nn.Conv2d(64, 64, 1)
        self.up_conv1x1_4 = nn.Conv2d(32, 32, 1)

        self.dconv_up4 = LocalNet(256, 128)
        self.dconv_up3 = LocalNet(192, 64)
        self.dconv_up2 = LocalNet(96, 32)
        self.dconv_up1 = LocalNet(48, 16)

        self.conv_last = LocalNet(16, 3)

    def forward(self, x):
        """UNet implementation

        :param x: image
        :returns: predicted image
        :rtype: Tensor

        """
        x_in_tile = x.clone()

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.up_conv1x1_1(self.upsample(x))

        if x.shape[3] != conv4.shape[3] and x.shape[2] != conv4.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv4.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv4.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.up_conv1x1_2(self.upsample(x))

        if x.shape[3] != conv3.shape[3] and x.shape[2] != conv3.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv3.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv3.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.up_conv1x1_3(self.upsample(x))

        del conv3

        if x.shape[3] != conv2.shape[3] and x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv2.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.up_conv1x1_4(self.upsample(x))

        del conv2

        if x.shape[3] != conv1.shape[3] and x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv1.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv1], dim=1)
        del conv1

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = out + x_in_tile

        return out


class LocalNet(nn.Module):

    def forward(self, x_in):
        """Double convolutional block

        :param x_in: image features
        :returns: image features
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        x = self.lrelu(self.conv2(self.refpad(x)))

        return x

    def __init__(self, in_channels=16, out_channels=64):
        """Double convolutional block

        :param in_channels:  number of input channels
        :param out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)


# Model definition
class UNetModel(nn.Module):

    def __init__(self):
        """UNet runtime definition

        :returns: N/A
        :rtype: N/A

        """

        super(UNetModel, self).__init__()

        self.unet = UNet()
        self.final_conv = nn.Conv2d(3, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, image):
        """UNet runtime definition

        :param image: input image
        :returns: image features
        :rtype: Tensor
        """

        output_image = self.unet(image)
        return self.final_conv(self.refpad(output_image))
