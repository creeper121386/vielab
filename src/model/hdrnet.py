import os.path as osp

import torch.optim as optim
import util
from globalenv import *

from .basemodel import BaseModel
from .basic_loss import LTVloss
from .basic_loss import L_TV, L_spa, L_color, L_exp

ONNX_INPUT_W = 960
ONNX_INPUT_H = 720


def debug_tensor(tensor, name):
    np.save(name, util.cuda_tensor_to_ndarray(tensor))


class HDRnetLitModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt, [TRAIN, VALID])
        # if opt[AUGMENTATION][DOWNSAMPLE] != [512, 512]:
        #     console.log(
        #         f'[yellow]HDRnet requires input image size must be [512, 512], However your augmentation mathod is: \n{opt[AUGMENTATION]}. \nMake sure you do the correct augmentation! (Press enter to ignore and continue running.)[/yellow]')
        #     input()

        self.net = HDRPointwiseNN(opt[RUNTIME])
        low_res = opt[RUNTIME][LOW_RESOLUTION]

        # for torch1.7:
        # self.down_sampler = Resize([low_res, low_res])

        # for torch 1.5:
        self.down_sampler = lambda x: F.interpolate(x, size=(low_res, low_res), mode='bicubic', align_corners=False)

        self.use_illu = opt[RUNTIME][PREDICT_ILLUMINATION]

        if not opt[RUNTIME][SELF_SUPERVISED]:
            self.mse = torch.nn.MSELoss()
            self.ltv = LTVloss()
            self.cos = torch.nn.CosineSimilarity(1, 1e-8)
        else:
            console.log('HDRnet in SELF_SUPERVISED mode. Use zerodce losses.')
            self.color_loss = L_color()
            self.spatial_loss = L_spa()
            self.exposure_loss = L_exp(16, 0.6)
            self.tvloss = L_TV()

        self.net.train()

    def configure_optimizers(self):
        # self.parameters in LitModel is the same as nn.Module.
        # once you add nn.xxxx as a member in __init__, self.parameters will include it.

        optimizer = optim.Adam(self.net.parameters(), lr=self.opt[LR])
        return optimizer

    def training_step(self, batch, batch_idx):
        input_batch, gt_batch, fpaths = batch[INPUT], batch[GT], batch[FPATH]
        low_res_batch = self.down_sampler(input_batch)
        output_batch = self.net(low_res_batch, input_batch)

        # supervised training (get loss from output and GT)
        if not self.opt[RUNTIME][SELF_SUPERVISED]:
            # calculate loss:
            mse_loss = self.mse(output_batch, gt_batch)
            loss = mse_loss
            logged_losses = {MSE: mse_loss}

            # add cos loss
            cos_weight = self.opt[RUNTIME][LOSS][COS_LOSS]
            if cos_weight:
                cos_loss = cos_weight * (1 - self.cos(output_batch, gt_batch).mean()) * 0.5
                loss += cos_loss
                logged_losses[COS_LOSS] = cos_loss

            # add ltv loss
            ltv_weight = self.opt[RUNTIME][LOSS][LTV_LOSS]
            if self.use_illu and ltv_weight:
                tv_loss = self.ltv(input_batch, self.net.illu_map, ltv_weight)
                loss += tv_loss
                logged_losses[LTV_LOSS] = tv_loss
            logged_losses[LOSS] = loss

        else:
            # use losses of zeroDCE to do self-supervised training:
            if self.use_illu:
                loss_tv = 0.2 * self.tvloss(self.net.illu_map)
            else:
                loss_tv = 0.2 * self.tvloss(self.net.slice_coeffs)
            loss_spatial = 0.5 * torch.mean(self.spatial_loss(output_batch, input_batch))
            loss_color = 0.2 * torch.mean(self.color_loss(output_batch))
            loss_exposure = 0.2 * torch.mean(self.exposure_loss(output_batch))
            loss = loss_tv + loss_spatial + loss_color + loss_exposure
            logged_losses = {
                LTV_LOSS: loss_tv,
                SPATIAL_LOSS: loss_spatial,
                COLOR_LOSS: loss_color,
                EXPOSURE_LOSS: loss_exposure,
                LOSS: loss
            }

        # logging:
        self.log_dict(logged_losses)
        self.log_images_dict(
            TRAIN,
            osp.basename(fpaths[0]),
            {
                INPUT: input_batch,
                OUTPUT: output_batch,
                GT: gt_batch,
                PREDICT_ILLUMINATION: self.net.illu_map,
                GUIDEMAP: self.net.guidemap
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.global_valid_step += 1
        input_batch, gt_batch, fname = batch[INPUT], batch[GT], batch[FPATH][0]
        low_res_batch = self.down_sampler(input_batch)
        output_batch = self.net(low_res_batch, input_batch)

        # log metrics
        # if self.global_valid_step % 100 == 0:
        psnr = util.ImageProcessing.compute_psnr(
            util.cuda_tensor_to_ndarray(output_batch),
            util.cuda_tensor_to_ndarray(gt_batch), 1.0
        )
        self.log(PSNR, psnr)

        # log images
        self.log_images_dict(
            VALID,
            osp.basename(fname),
            {
                INPUT: input_batch,
                OUTPUT: output_batch,
                GT: gt_batch,
                PREDICT_ILLUMINATION: self.net.illu_map,
                GUIDEMAP: self.net.guidemap
            }
        )
        return output_batch

    def test_step(self, batch, batch_ix):
        # test without GT image:
        input_batch, fname = batch[INPUT], batch[FPATH][0]
        assert input_batch.shape[0] == 1
        low_res_batch = self.down_sampler(input_batch)
        output_batch = self.net(low_res_batch, input_batch)
        self.save_one_img_of_batch(
            output_batch, self.opt[IMG_DIRPATH], osp.basename(fname))

        # test with GT:
        if GT in batch:
            # calculate metrics:
            output_ = util.cuda_tensor_to_ndarray(output_batch)
            y_ = util.cuda_tensor_to_ndarray(batch[GT])
            psnr = util.ImageProcessing.compute_psnr(output_, y_, 1.0)
            ssim = util.ImageProcessing.compute_ssim(output_, y_)
            self.log_dict({PSNR: psnr, SSIM: ssim}, prog_bar=True, on_step=True, on_epoch=True)

    def forward(self, x):
        low_res_x = self.down_sampler(x)
        return self.net(low_res_x, x)


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


# for exporting onnx only!
class FakeGridSampler(nn.Module):
    #  shape: [1, 12, 8, 16, 16]
    # guidemap shape: [1, 1, 720, 1280]
    # return shape: [1, 12, 1, 720, 1280]

    def forward(self, bilateral_grid, guidemap):
        # h, w = guidemap_guide.shape[2:4]
        # bs = guidemap_guide.shape[0]

        fake_returned_value = torch.ones([1, 12, 1, ONNX_INPUT_H, ONNX_INPUT_W]).type_as(guidemap)
        fake_returned_value /= guidemap
        fake_returned_value /= bilateral_grid[0, 0, 0, 0, 0]
        return fake_returned_value


class Slice(nn.Module):
    def __init__(self, opt):
        super(Slice, self).__init__()
        self.opt = opt
        if opt[ONNX_EXPORTING_MODE]:
            self.fake_grid_sampler = FakeGridSampler()

    def forward(self, bilateral_grid, guidemap):
        # bilateral_grid shape: Nx12x8x16x16
        device = bilateral_grid.get_device()
        import ipdb;
        ipdb.set_trace()

        if not self.opt[ONNX_EXPORTING_MODE]:
            N, _, H, W = guidemap.shape
            hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
            if device >= 0:
                hg = hg.to(device)
                wg = wg.to(device)

            # import ipdb;ipdb.set_trace()
            hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
            wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
            guidemap = guidemap * 2 - 1
            guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
            guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)

            # guidemap shape: [N, 1 (D), H, W]
            # bilateral_grid shape: [N, 12 (c), 8 (d), 16 (h), 16 (w)], which is considered as a 3D space: [8, 16, 16]
            # guidemap_guide shape: [N, 1 (D), H, W, 3], which is considered as a 3D space: [1, H, W]
            # coeff shape: [N, 12 (c), 1 (D), H, W]

            # in F.grid_sample, gird is guidemap_guide, input is bilateral_grid
            # guidemap_guide[N, D, H, W] is a 3-vector <x, y, z>. but:
            #       x -> W, y -> H, z -> D  in bilater_grid
            # What does it really do:
            #   [ 1 ] For pixel in guidemap_guide[D, H, W], get <x,y,z>, and:
            #   [ 2 ] Normalize <x, y, z> from [-1, 1] to [0, w - 1], [0, h - 1], [0, d - 1], respectively.
            #   [ 3 ] Locate pixel in bilateral_grid at position [N, :, z, y, x].
            #   [ 4 ] Interplate using the neighbor values as the output affine matrix.
            coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True)

        else:
            # >>>>>>>>> only for onnx exporting! <<<<<<<<<<<<<
            console.log('>>>>>>>>> [ FATAL WARN ] Use fake grid_sample! <<<<<<<<<<')
            coeff = self.fake_grid_sampler(bilateral_grid, guidemap)
            # >>>>>>>>> only for onnx exporting! <<<<<<<<<<<<<

        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        '''
        coeff shape: [bs, 12, h, w]
        input shape: [bs, 3, h, w]
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class ApplyCoeffsGamma(nn.Module):
    def __init__(self):
        super(ApplyCoeffsGamma, self).__init__()
        console.log('[ WARN ] Use alter methods indtead of affine matrix.')

    def forward(self, x_r, x):
        '''
        coeff shape: [bs, 12, h, w]
        apply zeroDCE curve.
        '''

        # [ 008 ] single iteration alpha map:
        # coeff channel num: 3
        # return x + x_r * (torch.pow(x, 2) - x)

        # [ 009 ] 8 iteratoins:
        # coeff channel num: 24
        # r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        # x = x + r1 * (torch.pow(x, 2) - x)
        # x = x + r2 * (torch.pow(x, 2) - x)
        # x = x + r3 * (torch.pow(x, 2) - x)
        # enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        # x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        # x = x + r6 * (torch.pow(x, 2) - x)
        # x = x + r7 * (torch.pow(x, 2) - x)
        # enhance_image = x + r8 * (torch.pow(x, 2) - x)
        # r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        # return enhance_image

        # [ 014 ] use illu map:
        # coeff channel num: 3
        # return x / (torch.where(x_r < x, x, x_r) + 1e-7)

        # [ 015 ] use HSV and only affine V channel:
        # coeff channel num: 3
        V = torch.sum(x * x_r, dim=1, keepdim=True) + x_r
        return torch.cat([x[:, 0:2, ...], V], dim=1)


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
        # print(n_layers_global)
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
        if not params[ONNX_EXPORTING_MODE]:
            bs = lowres_input.shape[0]
        else:
            # >>>>>>>>> only for onnx exporting! <<<<<<<<<<<<<
            console.log('>>>>>>>>> [ FATAL WARN ] Use fake batchsize for onnx! <<<<<<<<<<')
            bs = 1
            # >>>>>>>>> only for onnx exporting! <<<<<<<<<<<<<

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

        x = self.conv_out(fusion)

        # reshape channel dimension -> bilateral grid dimensions:
        # [bs, 96, 1, 1] -> [bs, 12, 8, 16, 16]
        y = torch.stack(torch.split(x, self.nin * self.nout, 1), 2)

        # y = torch.stack(torch.split(y, self.nin, 1),3)
        # print(y.shape)
        # x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        # print(x.shape)
        return y


class HDRPointwiseNN(nn.Module):

    def __init__(self, params):
        super(HDRPointwiseNN, self).__init__()
        self.opt = params
        self.guide = GuideNN(params=params)
        self.slice = Slice(params)
        if not params[SELF_SUPERVISED]:
            self.coeffs = Coeffs(params=params)
            self.apply_coeffs = ApplyCoeffs()
        else:
            console.log('HDRPointwiseNN in SELF_SUPERVISED mode.')

            # [ 008 ] change affine matrix -> other methods (alpha map, illu map)
            self.coeffs = Coeffs(params=params, nin=1)
            self.apply_coeffs = ApplyCoeffsGamma()

            # [ 011 ] still use affine matrix (same as supervised)
            # self.coeffs = Coeffs(params=params)
            # self.apply_coeffs = ApplyCoeffs()

    def forward(self, lowres, fullres):
        bilateral_grid = self.coeffs(lowres)
        guide = self.guide(fullres)
        self.guidemap = guide
        slice_coeffs = self.slice(bilateral_grid, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)

        # use illu map:
        self.slice_coeffs = slice_coeffs
        if self.opt[PREDICT_ILLUMINATION]:
            self.illu_map = out
            out = fullres / (torch.where(out < fullres, fullres, out) + 1e-7)
        else:
            self.illu_map = None

        return out

#########################################################################################################
