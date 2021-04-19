import os
import os.path as osp

import torch.optim as optim
from globalenv import *
from toolbox import util
from torchvision.transforms import Resize

from .basemodel import BaseModel
from .basic_loss import LTVloss


class HDRnetLitModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt, [TRAIN])
        if opt[AUGMENTATION][DOWNSAMPLE] != [512, 512]:
            console.log(
                f'[yellow]HDRnet requires input image size must be [512, 512], However your augmentation mathod is: \n{opt[AUGMENTATION]}. \nMake sure you do the correct augmentation! (Press enter to continue)[/yellow]')
            input()

        self.net = HDRPointwiseNN(opt[RUNTIME])
        self.losses = {
            L1_LOSS: 0,
            LTV_LOSS: 0,
            COS_LOSS: 0,
            LOSS: 0
        }
        self.down_sampler = Resize([opt[RUNTIME][LOW_RESOLUTION]])

        # self.illumination_dirpath = os.path.join(opt[LOG_DIRPATH], PREDICT_ILLUMINATION)
        # if opt[RUNTIME][PREDICT_ILLUMINATION]:
        #     util.mkdir(self.illumination_dirpath)

        self.mse = torch.nn.MSELoss()
        self.ltvloss = LTVloss()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)
        self.net.train()

    def configure_optimizers(self):
        # self.parameters in LitModel is the same as nn.Module.
        # once you add nn.xxxx as a member in __init__, self.parameters will include it.

        optimizer = optim.Adam(self.net.parameters(), lr=self.opt[LR])
        return optimizer

    def training_step(self, batch, batch_idx):
        # watch model gradients:
        # if not self.MODEL_WATCHED:
        #     self.logger.watch(self.net)
        #     self.MODEL_WATCHED = True

        input_batch, gt_batch, fpaths = batch[INPUT], batch[GT], batch[FPATH]
        low_res_batch = self.down_sampler(input_batch)
        output_batch = self.net(low_res_batch, input_batch)
        mse_loss = self.mse(output_dict, gt_batch)

        # todo: 本行以下代码还没改

        illumination = None if not self.opt[RUNTIME][PREDICT_ILLUMINATION] else output_dict[PREDICT_ILLUMINATION]
        self.training_step_logging(input_batch, gt_batch, output, fpaths, illumination)
        return loss

    def training_step_logging(self, input_batch, gt_batch, output, fpaths, illumination):
        '''
        ! logging AVERAGE value of self.losses of current step.
        '''

        # for logging loss:
        this_losses = self.criterion.get_current_loss()
        for k in self.losses:
            if this_losses[k] is not None:
                # TODO: 多卡训练时，这里报错两个tensor分别在两块卡上：
                self.losses[k] = this_losses[k]
            else:
                self.losses[k] = STRING_FALSE

        # log to the terminal and logger
        for x, y in self.losses.items():
            if y != STRING_FALSE:
                # `self.log_dict` will cause key error.
                self.log(x, y, prog_bar=True, on_step=True, on_epoch=False)

        # save images
        # note: self.global_step increases in training_step only, not effected by validation.
        if self.global_step % self.opt[LOG_EVERY] == 0:
            # TODO: 解决dlpf训练会崩的问题（猜测是crop size或batchsize的原因）
            # TODO: 添加unet
            fname = f'epoch{self.current_epoch}_iter{self.global_step}_{osp.basename(fpaths[0])}.png'
            self.logger_buffer_add_img(TRAIN, input_batch, TRAIN, INPUT, fname)
            self.logger_buffer_add_img(TRAIN, output, TRAIN, OUTPUT, fname)
            self.logger_buffer_add_img(TRAIN, gt_batch, TRAIN, GT, fname)

            self.save_one_img_of_batch(output, self.opt[IMG_DIRPATH], fname)

            # save illumination map
            if illumination is not None:
                self.logger_buffer_add_img(TRAIN, illumination, TRAIN, PREDICT_ILLUMINATION, fname)
                self.save_one_img_of_batch(illumination, self.illumination_dirpath, fname)

            self.commit_logger_buffer(TRAIN)

    def test_step(self, batch, batch_ix):
        # test without GT image:
        input_batch, fname = batch[INPUT], batch[FPATH][0]
        output_dict = self.net(input_batch)
        output = torch.clamp(output_dict[OUTPUT], 0.0, 1.0)

        util.saveTensorAsImg(output, os.path.join(self.opt[IMG_DIRPATH], osp.basename(fname)))
        if PREDICT_ILLUMINATION in output_dict:
            util.saveTensorAsImg(
                output_dict[PREDICT_ILLUMINATION],
                os.path.join(self.illumination_dirpath, osp.basename(fname))
            )

        # test with GT:
        if GT in batch:
            # calculate metrics:
            output_ = util.cuda_tensor_to_ndarray(output)
            y_ = util.cuda_tensor_to_ndarray(batch[GT])
            psnr = util.ImageProcessing.compute_psnr(output_, y_, 1.0)
            ssim = util.ImageProcessing.compute_ssim(output_, y_)
            self.log_dict({PSNR: psnr, SSIM: ssim}, prog_bar=True)

    def forward(self, x):
        return self.net(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        # Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3

        # >>> origin
        # import ipdb; ipdb.set_trace()
        # ???????这里为什么input, grid对应的是grid, guidemap啊
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True)

        # bilateral_grid shape: [1, 12, 8, 16, 16]
        # guidemap_guide shape: [1, 1, 1080, 1920, 3]
        # coeff shape: [1, 12, 1, 1080, 1920]

        # >>> our implementation
        # coeff = bilinear_interpolate_torch_2D(
        #    bilateral_grid, guidemap_guide, align_corners=True)

        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Sigmoid)  # nn.Tanh

    def forward(self, x):
        return self.conv2(self.conv1(x))  # .squeeze(1)


class Coeffs(nn.Module):

    def __init__(self, nin=4, nout=3, params=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin
        self.nout = nout

        lb = params[LUMA_BINS]
        cm = params[CHANNEL_MULTIPLIER]
        sb = params[SPATIAL_BIN]
        bn = params[BATCH_NORM]
        nsize = params[LOW_RESOLUTION]

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize / sb))
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm * (2 ** i) * lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm * (2 ** i) * lb

        # global features
        n_layers_global = int(np.log2(sb / 4))
        print(n_layers_global)
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(ConvBlock(prev_ch, cm * 8 * lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm * 8 * lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize / 2 ** n_total) ** 2
        self.global_features_fc.append(FC(prev_ch, 32 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(FC(32 * cm * lb, 16 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(FC(16 * cm * lb, 8 * cm * lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8 * cm * lb, 3, batch_norm=bn))
        self.local_features.append(ConvBlock(8 * cm * lb, 8 * cm * lb, 3, activation=None, use_bias=False))

        # predicton
        self.conv_out = ConvBlock(8 * cm * lb, lb * nout * nin, 1, padding=0, activation=None)

    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params[LUMA_BINS]
        cm = params[CHANNEL_MULTIPLIER]
        sb = params[SPATIAL_BIN]

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x

        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        # shape: bs x 64 x 16 x 16
        fusion_grid = local_features

        # shape: bs x 64 x 1 x 1
        fusion_global = global_features.view(bs, 8 * cm * lb, 1, 1)
        fusion = self.relu(fusion_grid + fusion_global)

        # import ipdb; ipdb.set_trace()

        x = self.conv_out(fusion)
        s = x.shape
        y = torch.stack(torch.split(x, self.nin * self.nout, 1), 2)
        # y = torch.stack(torch.split(y, self.nin, 1),3)
        # print(y.shape)
        # x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        # print(x.shape)
        return y


class HDRPointwiseNN(nn.Module):

    def __init__(self, params):
        super(HDRPointwiseNN, self).__init__()
        self.coeffs = Coeffs(params=params)
        self.guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        # self.bsa = bsa.BilateralSliceApply()

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        # out = bsa.bsa(coeffs,guide,fullres)

        # >>>Using illumination!<<
        import sys
        sys.stdout.write('\r>>>Using illumination!<<<')
        # import ipdb; ipdb.set_trace()
        self.illumination_map = out
        input = fullres
        out = input / (torch.where(out < input, input, out) + 1e-7)
        # >>>Using illumination!<<

        return out

#########################################################################################################
