import math
import os
import os.path as osp
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from globalenv import *
import util
from torch.autograd import Variable

from . import unet as unet
from .basemodel import BaseModel
from .basic_loss import LTVloss


class DeepLpfLitModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt, [TRAIN])
        if opt[VALID_DATA][INPUT]:
            console.log('[yellow]WARNING: valid mode is not supported in DeepLPF, ignore valid_dataloader.[/yellow]')

        self.net = DeepLPFNet(opt)
        self.losses = {
            SSIM_LOSS: 0,
            L1_LOSS: 0,
            LTV_LOSS: 0,
            COS_LOSS: 0,
            LOSS: 0
        }

        self.metrics = {
            PSNR: 0,
            SSIM: 0
        }
        self.illumination_dirpath = os.path.join(opt[LOG_DIRPATH], PREDICT_ILLUMINATION)
        if opt[RUNTIME][PREDICT_ILLUMINATION]:
            util.mkdir(self.illumination_dirpath)

        self.criterion = DeepLPFLoss(opt, ssim_window_size=5)
        self.net.train()

    def configure_optimizers(self):
        # self.parameters in LitModel is the same as nn.Module.
        # once you add nn.xxxx as a member in __init__, self.parameters will include it.

        optimizer = optim.Adam(self.net.parameters(), lr=self.opt[LR], betas=(0.9, 0.999), eps=1e-08)
        return optimizer

    def training_step(self, batch, batch_idx):
        # watch model gradients:
        # if not self.MODEL_WATCHED:
        #     self.logger.watch(self.net)
        #     self.MODEL_WATCHED = True

        input_batch, gt_batch, fpaths = Variable(batch[INPUT], requires_grad=False), \
                                        Variable(batch[GT],
                                                 requires_grad=False), batch[FPATH]
        output_dict = self.net(input_batch)
        output = torch.clamp(output_dict[OUTPUT], 0.0, 1.0)

        loss = self.criterion(output_dict, gt_batch)

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
            fname = f'epoch{self.current_epoch}_iter{self.global_step}_{osp.basename(fpaths[0])}.png'
            self.add_img_to_buffer(TRAIN, input_batch, TRAIN, INPUT, fname)
            self.add_img_to_buffer(TRAIN, output, TRAIN, OUTPUT, fname)
            self.add_img_to_buffer(TRAIN, gt_batch, TRAIN, GT, fname)

            self.save_one_img_of_batch(output, self.opt[IMG_DIRPATH], fname)

            # save illumination map
            if illumination is not None:
                self.add_img_to_buffer(TRAIN, illumination, TRAIN, PREDICT_ILLUMINATION, fname)
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


class DeepLPFLoss(nn.Module):

    def __init__(self, opt, ssim_window_size=5, alpha=0.5):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFLoss, self).__init__()
        self.alpha = alpha
        self.opt = opt
        self.ssim_window_size = ssim_window_size
        self.cos = nn.CosineSimilarity(1, 1e-8)
        self.ltv = LTVloss()
        self.losses = {
            SSIM_LOSS: None,
            L1_LOSS: None,
            LTV_LOSS: None,
            COS_LOSS: None,
            LOSS: None
        }

        assert LTV_LOSS in self.opt[RUNTIME][LOSS]
        assert COS_LOSS in self.opt[RUNTIME][LOSS]

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        assert img1.type() == img2.type()

        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.type_as(img1) + mu2_sq.type_as(img1) + C1) *
                     (sigma1_sq.type_as(img1) + sigma2_sq.type_as(img1) + C2))
        ssim_map = ssim_map1.type_as(img1) / ssim_map2.type_as(img1)

        v1 = 2.0 * sigma12.type_as(img1) + C2
        v2 = sigma1_sq.type_as(img1) + sigma2_sq.type_as(img1) + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs

    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.size() != img2.size():
            raise RuntimeError('Input images must have the same shape (%s vs. %s).' % (
                img1.size(), img2.size()))
        if len(img1.size()) != 4:
            raise RuntimeError(
                'Input images must have four dimensions, not %d' % len(img1.size()))

        '''
        if type(img1) is not Variable or type(img2) is not Variable:
            raise RuntimeError(
                'Input images must be Variables, not %s' % img1.__class__.__name__)
        '''

        weights = Variable(torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        # weights = Variable(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0]))
        if img1.is_cuda:
            weights = weights.type_as(img1)

        levels = weights.size()[0]
        # print(levels)
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.compute_ssim(img1, img2)
            mssim.append(sim.unsqueeze(dim=0))
            mcs.append(cs.unsqueeze(dim=0))

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        # print(mssim)
        img1 = img1.contiguous()
        img2 = img2.contiguous()

        mssim = torch.cat(mssim)
        mcs = torch.cat(mcs)

        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

        prod = (torch.prod(mcs[0:levels - 1] ** weights[0:levels - 1])
                * (mssim[levels - 1] ** weights[levels - 1]))
        return prod

    def forward(self, output_dict, gt_batch):
        """Forward function for the DeepLPF loss

        :param output_dict: output dict with key OUTPUT and 'illumination'(optional) for network
        :param gt_batch: Tensor of shape BxCxWxH
        :returns: value of loss function
        :rtype: float

        """
        assert OUTPUT in output_dict and INPUT in output_dict

        predicted_img_batch = output_dict[OUTPUT]
        num_images = gt_batch.shape[0]

        assert predicted_img_batch.type() == gt_batch.type()

        ssim_loss_value = torch.zeros(1, 1).type_as(predicted_img_batch)
        l1_loss_value = torch.zeros(1, 1).type_as(predicted_img_batch)

        for i in range(0, num_images):
            target_img = gt_batch[i, :, :, :].type_as(predicted_img_batch)
            predicted_img = predicted_img_batch[i, :, :, :].type_as(predicted_img_batch)

            predicted_img_lab = util.ImageProcessing.rgb_to_lab(
                predicted_img.squeeze(0))
            target_img_lab = util.ImageProcessing.rgb_to_lab(target_img.squeeze(0))

            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            # SSIM loss
            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)
            ssim_loss_value += (1.0 - ssim_value)

            # L1 loss
            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)

        l1_loss_value = l1_loss_value / num_images
        ssim_loss_value = ssim_loss_value / num_images

        deeplpf_loss = l1_loss_value + 1e-3 * ssim_loss_value

        self.losses[L1_LOSS] = l1_loss_value
        self.losses[SSIM_LOSS] = ssim_loss_value

        # ─── LOCAL SMOOTHNESS LOSS ───────────────────────────────────────
        ltvWeight = self.opt[RUNTIME][LOSS][LTV_LOSS]

        if ltvWeight:
            assert type(ltvWeight + 0.1) == float

            if self.opt[RUNTIME][PREDICT_ILLUMINATION]:
                # apply ltv loss on illumination:
                assert PREDICT_ILLUMINATION in output_dict
                smoothed_map_name = PREDICT_ILLUMINATION
            else:
                smoothed_map_name = OUTPUT

            self.losses[LTV_LOSS] = self.ltv(
                output_dict[INPUT], output_dict[smoothed_map_name], ltvWeight)
            deeplpf_loss += self.losses[LTV_LOSS]

        # ─── COS SIMILARITY ──────────────────────────────────────────────
        cosWeight = self.opt[RUNTIME][LOSS][COS_LOSS]
        if cosWeight:
            assert type(cosWeight + 0.1) == float

            cos_loss = self.cos(predicted_img_batch,
                                gt_batch).mean() * cosWeight
            self.losses[COS_LOSS] = cos_loss
            deeplpf_loss += cos_loss
        # ─────────────────────────────────────────────────────────────────

        self.losses[LOSS] = deeplpf_loss
        return deeplpf_loss

    def get_current_loss(self):
        return self.losses


class BinaryLayer(nn.Module):

    def forward(self, input):
        """Forward function for binary layer

        :param input: data
        :returns: sign of data
        :rtype: Tensor

        """
        return torch.sign(input)

    def backward(self, grad_output):
        """Straight through estimator

        :param grad_output: gradient tensor
        :returns: truncated gradient tensor
        :rtype: Tensor

        """
        input = self.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class CubicFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CubicFilter, self).__init__()

        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.cubic_layer2 = MaxPoolBlock()
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer4 = MaxPoolBlock()
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer6 = MaxPoolBlock()
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer8 = GlobalPoolingBlock(2)
        self.fc_cubic = torch.nn.Linear(
            num_out_channels, 60)  # cubic
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.5)

    def get_cubic_mask(self, feat, img):
        """Cubic filter definition

        :param feat: feature map
        :param img:  image
        :returns: cubic scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Cubic #########################
        feat_cubic = torch.cat((feat, img), 1)
        feat_cubic = self.upsample(feat_cubic)

        x = self.cubic_layer1(feat_cubic)
        x = self.cubic_layer2(x)
        x = self.cubic_layer3(x)
        x = self.cubic_layer4(x)
        x = self.cubic_layer5(x)
        x = self.cubic_layer6(x)
        x = self.cubic_layer7(x)
        x = self.cubic_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        # R is the paramters predicted:
        R = self.fc_cubic(x)

        cubic_mask = torch.zeros_like(img)

        x_axis = Variable(torch.arange(
            img.shape[2]).view(-1, 1).repeat(1, img.shape[3]).type_as(img)) / img.shape[2]
        y_axis = Variable(torch.arange(img.shape[3]).repeat(
            img.shape[2], 1).type_as(img)) / img.shape[3]

        x_axis = x_axis.float()
        y_axis = y_axis.float()

        # generate specific filter using R:
        '''
        Cubic for R channel
        '''
        cubic_mask[0, 0, :, :] = R[0, 0] * (x_axis ** 3) + R[0, 1] * (x_axis ** 2) * y_axis + R[0, 2] * (
                x_axis ** 2) * img[0, 0, :, :] + R[0, 3] * (x_axis ** 2) + R[0, 4] * x_axis * (y_axis ** 2) + R[
                                     0, 5] * x_axis * y_axis * img[0, 0, :, :] \
                                 + R[0, 6] * x_axis * y_axis + R[0, 7] * x_axis * (img[0, 0, :, :] ** 2) + R[
                                     0, 8] * x_axis * img[0, 0, :, :] + R[0, 9] * x_axis + R[0, 10] * (
                                         y_axis ** 3) + R[0, 11] * (y_axis ** 2) * img[0, 0, :, :] \
                                 + R[0, 12] * (y_axis ** 2) + R[0, 13] * y_axis * (img[0, 0, :, :] ** 2) + R[
                                     0, 14] * y_axis * img[0, 0, :, :] + R[0, 15] * y_axis + R[0, 16] * (
                                         img[0, 0, :, :] ** 3) + R[0, 17] * (img[0, 0, :, :] ** 2) \
                                 + R[0, 18] * \
                                 img[0, 0, :, :] + R[0, 19]

        '''
        Cubic for G channel
        '''
        cubic_mask[0, 1, :, :] = R[0, 20] * (x_axis ** 3) + R[0, 21] * (x_axis ** 2) * y_axis + R[0, 22] * (
                x_axis ** 2) * img[0, 1, :, :] + R[0, 23] * (x_axis ** 2) + R[0, 24] * x_axis * (y_axis ** 2) + R[
                                     0, 25] * x_axis * y_axis * img[0, 1, :, :] \
                                 + R[0, 26] * x_axis * y_axis + R[0, 27] * x_axis * (img[0, 1, :, :] ** 2) + R[
                                     0, 28] * x_axis * img[0, 1, :, :] + R[0, 29] * x_axis + R[0, 30] * (
                                         y_axis ** 3) + R[0, 31] * (y_axis ** 2) * img[0, 1, :, :] \
                                 + R[0, 32] * (y_axis ** 2) + R[0, 33] * y_axis * (img[0, 1, :, :] ** 2) + R[
                                     0, 34] * y_axis * img[0, 1, :, :] + R[0, 35] * y_axis + R[0, 36] * (
                                         img[0, 1, :, :] ** 3) + R[0, 37] * (img[0, 1, :, :] ** 2) \
                                 + R[0, 38] * \
                                 img[0, 1, :, :] + R[0, 39]

        '''
        Cubic for B channel
        '''
        cubic_mask[0, 2, :, :] = R[0, 40] * (x_axis ** 3) + R[0, 41] * (x_axis ** 2) * y_axis + R[0, 42] * (
                x_axis ** 2) * img[0, 2, :, :] + R[0, 43] * (x_axis ** 2) + R[0, 44] * x_axis * (y_axis ** 2) + R[
                                     0, 45] * x_axis * y_axis * img[0, 2, :, :] \
                                 + R[0, 46] * x_axis * y_axis + R[0, 47] * x_axis * (img[0, 2, :, :] ** 2) + R[
                                     0, 48] * x_axis * img[0, 2, :, :] + R[0, 49] * x_axis + R[0, 50] * (
                                         y_axis ** 3) + R[0, 51] * (y_axis ** 2) * img[0, 2, :, :] \
                                 + R[0, 52] * (y_axis ** 2) + R[0, 53] * y_axis * (img[0, 2, :, :] ** 2) + R[
                                     0, 54] * y_axis * img[0, 2, :, :] + R[0, 55] * y_axis + R[0, 56] * (
                                         img[0, 2, :, :] ** 3) + R[0, 57] * (img[0, 2, :, :] ** 2) \
                                 + R[0, 58] * \
                                 img[0, 2, :, :] + R[0, 59]

        img_cubic = torch.clamp(img + cubic_mask, 0, 1)
        return img_cubic  # shape (1, 3, h, w)


class GraduatedFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialization function for the graduated filter

        :param num_in_channels:  input channels
        :param num_out_channels: output channels
        :returns: N/A
        :rtype: N/A

        """
        super(GraduatedFilter, self).__init__()

        self.graduated_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.graduated_layer2 = MaxPoolBlock()
        self.graduated_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer4 = MaxPoolBlock()
        self.graduated_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer6 = MaxPoolBlock()
        self.graduated_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer8 = GlobalPoolingBlock(2)
        self.fc_graduated = torch.nn.Linear(
            num_out_channels, 24)
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.5)
        self.bin_layer = BinaryLayer()

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x.float()) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors

        :param cond: condition e.g. <
        :param x_1: Tensor 1
        :param x_2: Tensor 2
        :returns: Boolean comparison result
        :rtype: Tensor

        """
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_inverted_mask(self, factor, invert, d1, d2, max_scale, top_line):
        """ Inverts the graduated filter based on a learnt binary variable

        :param factor: scale factor
        :param invert: binary indicator variable
        :param d1: distance between top and mid line
        :param d2: distannce between botto and mid line
        :param max_scale: maximum scaling factor possible
        :param top_line:  representation of top line
        :returns: inverted scaling mask
        :rtype: Tensor

        """
        if (invert == 1).all():

            if (factor >= 1).all():
                diff = ((factor - 1)) / 2 + 1
                grad1 = (diff - factor) / d1
                grad2 = (1 - diff) / d2
                mask_scale = torch.clamp(
                    factor + grad1 * top_line + grad2 * top_line, min=1, max=max_scale)
            else:
                diff = ((1 - factor)) / 2 + factor
                grad1 = (diff - factor) / d1
                grad2 = (1 - diff) / d2
                mask_scale = torch.clamp(
                    factor + grad1 * top_line + grad2 * top_line, min=0, max=1)
        else:

            if (factor >= 1).all():
                diff = ((factor - 1)) / 2 + 1
                grad1 = (diff - factor) / d1
                grad2 = (factor - diff) / d2
                mask_scale = torch.clamp(
                    1 + grad1 * top_line + grad2 * top_line, min=1, max=max_scale)
            else:
                diff = ((1 - factor)) / 2 + factor
                grad1 = (diff - 1) / d1
                grad2 = (factor - diff) / d2
                mask_scale = torch.clamp(
                    1 + grad1 * top_line + grad2 * top_line, min=0, max=1)

        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)
        return mask_scale

    def get_graduated_mask(self, feat, img):
        """ Graduated filter definition

        :param feat: features
        :param img: image
        :returns: scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Graduated #####################

        x_axis = Variable(torch.arange(
            img.shape[2]).view(-1, 1).repeat(1, img.shape[3]).type_as(img)) / img.shape[2]
        y_axis = Variable(torch.arange(img.shape[3]).repeat(
            img.shape[2], 1).type_as(img)) / img.shape[3]
        x_axis = x_axis.float()
        y_axis = y_axis.float()

        feat_graduated = torch.cat((feat, img), 1)
        feat_graduated = self.upsample(feat_graduated)

        x = self.graduated_layer1(feat_graduated)
        x = self.graduated_layer2(x)
        x = self.graduated_layer3(x)
        x = self.graduated_layer4(x)
        x = self.graduated_layer5(x)
        x = self.graduated_layer6(x)
        x = self.graduated_layer7(x)
        x = self.graduated_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_graduated(x)

        # Classification values (above or below the line)
        G[0, 21] = ((self.bin_layer(G[0, 21])) + 1) / 2
        G[0, 22] = ((self.bin_layer(G[0, 22])) + 1) / 2
        G[0, 23] = ((self.bin_layer(G[0, 23])) + 1) / 2

        slope1 = G[0, 0].clone()
        slope2 = G[0, 6].clone()
        slope3 = G[0, 12].clone()

        G[0, 1] = self.tanh01(G[0, 1]) + 1e-10
        G[0, 7] = self.tanh01(G[0, 7]) + 1e-10
        G[0, 13] = self.tanh01(G[0, 13]) + 1e-10

        G[0, 2] = torch.clamp(self.tanh01(G[0, 2]), G[0, 1].item(), 1.0)
        G[0, 8] = torch.clamp(self.tanh01(G[0, 8]), G[0, 7].item(), 1.0)
        G[0, 14] = torch.clamp(self.tanh01(G[0, 14]), G[0, 13].item(), 1.0)

        G[0, 18] = torch.clamp(self.tanh01(G[0, 18]), 0, G[0, 1].item())
        G[0, 19] = torch.clamp(self.tanh01(G[0, 19]), 0, G[0, 7].item())
        G[0, 20] = torch.clamp(self.tanh01(G[0, 20]), 0, G[0, 13].item())

        # Scales
        max_scale = 2
        min_scale = 0

        G[0, 3] = self.tanh01(G[0, 3]) * max_scale
        G[0, 4] = self.tanh01(G[0, 4]) * max_scale
        G[0, 5] = self.tanh01(G[0, 5]) * max_scale

        G[0, 9] = self.tanh01(G[0, 9]) * max_scale
        G[0, 10] = self.tanh01(G[0, 10]) * max_scale
        G[0, 11] = self.tanh01(G[0, 11]) * max_scale

        G[0, 15] = self.tanh01(G[0, 15]) * max_scale
        G[0, 16] = self.tanh01(G[0, 16]) * max_scale
        G[0, 17] = self.tanh01(G[0, 17]) * max_scale

        slope1_angle = torch.atan(slope1)
        slope2_angle = torch.atan(slope2)
        slope3_angle = torch.atan(slope3)

        d1 = self.tanh01(G[0, 2] * torch.cos(slope1_angle))
        d2 = self.tanh01(G[0, 18] * torch.cos(slope1_angle))
        d3 = self.tanh01(G[0, 8] * torch.cos(slope2_angle))
        d4 = self.tanh01(G[0, 19] * torch.cos(slope2_angle))
        d5 = self.tanh01(G[0, 14] * torch.cos(slope3_angle))
        d6 = self.tanh01(G[0, 20] * torch.cos(slope3_angle))

        top_line1 = self.tanh01(y_axis - (slope1 * x_axis + G[0, 1] + d1))
        top_line2 = self.tanh01(y_axis - (slope2 * x_axis + G[0, 7] + d3))
        top_line3 = self.tanh01(y_axis - (slope3 * x_axis + G[0, 13] + d5))

        mask_scale1 = self.get_inverted_mask(
            G[0, 3], G[0, 21], d1, d2, max_scale, top_line1)
        mask_scale2 = self.get_inverted_mask(
            G[0, 4], G[0, 21], d1, d2, max_scale, top_line1)
        mask_scale3 = self.get_inverted_mask(
            G[0, 5], G[0, 21], d1, d2, max_scale, top_line1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1 = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        mask_scale4 = self.get_inverted_mask(
            G[0, 9], G[0, 22], d3, d4, max_scale, top_line2)
        mask_scale5 = self.get_inverted_mask(
            G[0, 10], G[0, 22], d3, d4, max_scale, top_line2)
        mask_scale6 = self.get_inverted_mask(
            G[0, 11], G[0, 22], d3, d4, max_scale, top_line2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4 = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        mask_scale7 = self.get_inverted_mask(
            G[0, 15], G[0, 23], d5, d6, max_scale, top_line3)
        mask_scale8 = self.get_inverted_mask(
            G[0, 16], G[0, 23], d5, d6, max_scale, top_line3)
        mask_scale9 = self.get_inverted_mask(
            G[0, 17], G[0, 23], d5, d6, max_scale, top_line3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7 = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        mask_scale = torch.clamp(
            mask_scale_1 * mask_scale_4 * mask_scale_7, 0, max_scale)

        return mask_scale


class EllipticalFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(EllipticalFilter, self).__init__()

        self.elliptical_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.elliptical_layer2 = MaxPoolBlock()
        self.elliptical_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer4 = MaxPoolBlock()
        self.elliptical_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer6 = MaxPoolBlock()
        self.elliptical_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer8 = GlobalPoolingBlock(2)
        self.fc_elliptical = torch.nn.Linear(
            num_out_channels, 24)  # elliptical
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.5)

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors

        :param cond: condition e.g. <
        :param x_1: Tensor 1
        :param x_2: Tensor 2
        :returns: Boolean comparison result
        :rtype: Tensor

        """
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_mask(self, x_axis, y_axis, shift_x=0, shift_y=0, semi_axis_x=1, semi_axis_y=1, alpha=0,
                 scale_factor=2, max_scale=2, eps=1e-8, radius=1):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :returns: scaling mask
        :rtype: Tensor

        """
        mask_scale = self.where((((((x_axis - shift_x) * torch.cos(alpha) + (y_axis - shift_y) * torch.sin(
            alpha)) ** 2) / ((semi_axis_x) ** 2)) + ((((x_axis - shift_x) * torch.sin(alpha) - (
                y_axis - shift_y) * torch.cos(alpha)) ** 2) / ((semi_axis_y) ** 2)) + eps) < 1,
                                (torch.sqrt((x_axis - shift_x) ** 2 + (y_axis - shift_y) ** 2 + eps) * (
                                        1 - scale_factor)) / radius + scale_factor, 1)
        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)

        return mask_scale

    def get_elliptical_mask(self, feat, img):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :param feat: features from the backbone
        :param img: image
        :returns: elliptical adjustment maps for each channel
        :rtype: Tensor

        """
        eps = 1e-10
        max_scale = 2
        min_scale = 0

        feat_elliptical = torch.cat((feat, img), 1)
        feat_elliptical = self.upsample(feat_elliptical)

        x = self.elliptical_layer1(feat_elliptical)
        x = self.elliptical_layer2(x)
        x = self.elliptical_layer3(x)
        x = self.elliptical_layer4(x)
        x = self.elliptical_layer5(x)
        x = self.elliptical_layer6(x)
        x = self.elliptical_layer7(x)
        x = self.elliptical_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_elliptical(x)

        x_axis = Variable(torch.arange(
            img.shape[2]).view(-1, 1).repeat(1, img.shape[3]).type_as(img)) / img.shape[2]
        y_axis = Variable(torch.arange(img.shape[3]).repeat(
            img.shape[2], 1).type_as(img)) / img.shape[3]

        x_axis = x_axis.float()
        y_axis = y_axis.float()

        # x coordinate - h position
        right_x = (img.shape[2] - 1) / img.shape[2]
        left_x = 0

        G[0, 0] = self.tanh01(G[0, 0]) + eps
        G[0, 8] = self.tanh01(G[0, 8]) + eps
        G[0, 16] = self.tanh01(G[0, 16]) + eps

        # y coordinate - k coordinate
        right_y = (img.shape[3] - 1) // img.shape[3]
        left_y = 0

        G[0, 1] = self.tanh01(G[0, 1]) + eps
        G[0, 9] = self.tanh01(G[0, 9]) + eps
        G[0, 17] = self.tanh01(G[0, 17]) + eps

        # a value
        G[0, 2] = self.tanh01(G[0, 2]) + eps
        G[0, 10] = self.tanh01(G[0, 10]) + eps
        G[0, 18] = self.tanh01(G[0, 18]) + eps

        # b value
        G[0, 3] = self.tanh01(G[0, 3]) + eps  # * (right_x - left_x) + left_x
        G[0, 11] = self.tanh01(G[0, 11]) + eps
        G[0, 19] = self.tanh01(G[0, 19]) + eps

        # A value
        G[0, 4] = self.tanh01(G[0, 4]) * math.pi + eps
        G[0, 12] = self.tanh01(G[0, 12]) * math.pi + eps
        G[0, 20] = self.tanh01(G[0, 20]) * math.pi + eps

        '''
        The following are the scale factors for each ellipse
        '''
        G[0, 5] = self.tanh01(G[0, 5]) * max_scale + eps
        G[0, 6] = self.tanh01(G[0, 6]) * max_scale + eps
        G[0, 7] = self.tanh01(G[0, 7]) * max_scale + eps

        G[0, 13] = self.tanh01(G[0, 13]) * max_scale + eps
        G[0, 14] = self.tanh01(G[0, 14]) * max_scale + eps
        G[0, 15] = self.tanh01(G[0, 15]) * max_scale + eps

        G[0, 21] = self.tanh01(G[0, 21]) * max_scale + eps
        G[0, 22] = self.tanh01(G[0, 22]) * max_scale + eps
        G[0, 23] = self.tanh01(G[0, 23]) * max_scale + eps

        angle_1 = torch.acos(torch.clamp((y_axis - G[0, 1]) / (torch.sqrt(
            (x_axis - G[0, 0]) ** 2 + (y_axis - G[0, 1]) ** 2 + eps) + eps), -1 + 1e-7, 1 - 1e-7)) - G[0, 4]
        angle_2 = torch.acos(torch.clamp((y_axis - G[0, 9]) / (torch.sqrt((x_axis - G[0, 8]) ** 2 + (
                y_axis - G[0, 9]) ** 2 + eps) + eps), -1 + 1e-7, 1 - 1e-7)) - G[0, 12]
        angle_3 = torch.acos(torch.clamp((y_axis - G[0, 17]) / (torch.sqrt((x_axis - G[0, 16]) ** 2 + (
                y_axis - G[0, 17]) ** 2 + eps) + eps), -1 + 1e-7, 1 - 1e-7)) - G[0, 20]

        radius_1 = ((G[0, 2] * G[0, 3]) / (torch.sqrt((G[0, 2] ** 2) * (torch.sin(angle_1)
                                                                        ** 2) + (G[0, 3] ** 2) * (
                                                              torch.cos(angle_1) ** 2) + eps) + eps)) + eps
        radius_2 = ((G[0, 10] * G[0, 11]) / (torch.sqrt((G[0, 10] ** 2) * (torch.sin(angle_2)
                                                                           ** 2) + (G[0, 11] ** 2) * (
                                                                torch.cos(angle_2) ** 2) + eps) + eps)) + eps
        radius_3 = ((G[0, 18] * G[0, 19]) / (torch.sqrt((G[0, 18] ** 2) * (torch.sin(angle_3)
                                                                           ** 2) + (G[0, 19] ** 2) * (
                                                                torch.cos(angle_3) ** 2) + eps) + eps)) + eps

        mask_scale1 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 0], shift_y=G[0, 1], semi_axis_x=G[0,
                                                                                    2], semi_axis_y=G[0, 3],
                                    alpha=G[0, 4], scale_factor=G[0, 5],
                                    radius=radius_1)

        mask_scale2 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 0], shift_y=G[0, 1], semi_axis_x=G[0,
                                                                                    2], semi_axis_y=G[0, 3],
                                    alpha=G[0, 4], scale_factor=G[0, 6],
                                    radius=radius_1)

        mask_scale3 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 0], shift_y=G[0, 1], semi_axis_x=G[0,
                                                                                    2], semi_axis_y=G[0, 3],
                                    alpha=G[0, 4], scale_factor=G[0, 7],
                                    radius=radius_1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1_rad = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        ############

        mask_scale4 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 8], shift_y=G[0, 9], semi_axis_x=G[0,
                                                                                    10], semi_axis_y=G[0, 11],
                                    alpha=G[0, 12], scale_factor=G[0, 13],
                                    radius=radius_2)

        mask_scale5 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 8], shift_y=G[0, 9], semi_axis_x=G[0,
                                                                                    10], semi_axis_y=G[0, 11],
                                    alpha=G[0, 12], scale_factor=G[0, 14],
                                    radius=radius_2)

        mask_scale6 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 8], shift_y=G[0, 9], semi_axis_x=G[0,
                                                                                    10], semi_axis_y=G[0, 11],
                                    alpha=G[0, 12], scale_factor=G[0, 15],
                                    radius=radius_2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4_rad = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        ############

        mask_scale7 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 16], shift_y=G[0, 17], semi_axis_x=G[0,
                                                                                      18], semi_axis_y=G[0, 19],
                                    alpha=G[0, 20], scale_factor=G[0, 21],
                                    radius=radius_3)

        mask_scale8 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 16], shift_y=G[0, 17], semi_axis_x=G[0,
                                                                                      18], semi_axis_y=G[0, 19],
                                    alpha=G[0, 20], scale_factor=G[0, 22],
                                    radius=radius_3)

        mask_scale9 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 16], shift_y=G[0, 17], semi_axis_x=G[0,
                                                                                      18], semi_axis_y=G[0, 19],
                                    alpha=G[0, 20], scale_factor=G[0, 23],
                                    radius=radius_3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7_rad = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        mask_scale_elliptical = torch.clamp(
            mask_scale_1_rad * mask_scale_4_rad * mask_scale_7_rad, 0, max_scale)

        return mask_scale_elliptical


class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        input -> conv3x3 -> LeakeyRELU -> output

        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


class DeepLPFParameterPrediction(nn.Module):

    def __init__(self, opt, num_in_channels=64, num_out_channels=64):
        """Initialisation function

        :param num_in_channels:  Number of input feature maps
        :param num_out_channels: Number of output feature maps
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFParameterPrediction, self).__init__()
        self.opt = opt
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.cubic_filter = CubicFilter()
        self.graduated_filter = GraduatedFilter()
        self.elliptical_filter = EllipticalFilter()

    def forward(self, x):
        """DeepLPF combined architecture fusing cubic, graduated and elliptical filters

        :param x: forward the data Tensor x through the network
        :returns: Tensor representing the predicted image batch of shape BxCxWxH
        :rtype: Tensor

        """
        x.contiguous()  # remove memory holes

        feat = x[:, 3:64, :, :]
        img = x[:, 0:3, :, :]

        # torch.cuda.empty_cache()
        # shape = x.shape
        # img_clamped = torch.clamp(img, 0, 1)

        # "mask" means filter(image).
        img_cubic = self.cubic_filter.get_cubic_mask(feat, img)

        '''
        img     ---poly->           img_cubic
        img     ---graduated->      mask_scale_graduated
        img     ---elliptical->     mask_scale_elliptical
        (mask_scale_graduated + mask_scale_elliptical) * img_cubic --> res
        '''

        # Y3 = Y2 * (mask_scale_elliptical + mask_scale_graduated) :
        use_e = self.opt[RUNTIME][FILTERS][USE_ELLIPTICAL_FILTER]
        use_g = self.opt[RUNTIME][FILTERS][USE_GRADUATED_FILTER]

        if not use_e and not use_g:
            # no need for fusion
            res_residual = torch.clamp(img_cubic, 0, 1)
        else:
            mask_scale_graduated = self.graduated_filter.get_graduated_mask(
                feat, img)
            mask_scale_elliptical = self.elliptical_filter.get_elliptical_mask(
                feat, img)

            if use_e and use_g:
                mask_scale_fuse = torch.clamp(
                    mask_scale_graduated + mask_scale_elliptical, 0, 2)

            elif use_e and not use_g:
                mask_scale_fuse = torch.clamp(mask_scale_elliptical, 0, 1)
            elif use_g and not use_e:
                mask_scale_fuse = torch.clamp(mask_scale_graduated, 0, 1)

            res_residual = torch.clamp(img_cubic * mask_scale_fuse, 0, 1)

        # Y = Y1 + Y3 :
        img = torch.clamp(res_residual + img, 0, 1)

        return img


class DeepLPFNet(nn.Module):

    def __init__(self, opt):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(DeepLPFNet, self).__init__()
        self.opt = opt
        self.backbonenet = unet.UNetModel()
        self.deeplpfnet = DeepLPFParameterPrediction(opt)

    def forward(self, input):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        feat = self.backbonenet(input)
        output = self.deeplpfnet(feat) + 1e-8
        output_dict = dict()

        if self.opt[RUNTIME][PREDICT_ILLUMINATION]:
            output_dict[PREDICT_ILLUMINATION] = output

            # limit (input / out) to 0 ~ 1
            # 对于输出结果(L)中，比input还小的部分，直接取input
            # 防止预测的L，比input值（暗光图）还要小:
            output = input / (torch.where(output < input, input, output) / 1e-7)

        output_dict[INPUT] = input
        output_dict[OUTPUT] = output
        return output_dict
