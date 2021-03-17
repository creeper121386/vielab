import itertools
import os.path as osp

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import trilinear
from globalenv import *
from torch.autograd import Variable
from util import saveTensorAsImg, ImageProcessing


class IA3DLUTLitModel(pl.core.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        self.luts = [Generator3DLUT_identity(opt), Generator3DLUT_zero(), Generator3DLUT_zero()]
        self.cnn = Classifier()
        self.tv3 = TV_3D()
        self.trilinear = TrilinearInterpolation()

        self.train_img_dirpath = osp.join(opt[IMG_DIRPATH], TRAIN)
        self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)

        self.iternum = 0
        self.epoch = 0
        self.opt = opt
        self.criterion = torch.nn.MSELoss()

        self.train_metrics = {
            # PSNR: 0,
            MSE: 0,  # mse loss
            TV_CONS: 0,
            MN_CONS: 0,
            WEIGHTS_NORM: 0,  # L2 norm of output of cnn.
            LOSS: 0  # training loss
        }

        self.valid_metrics = {
            PSNR: 0,
            WEIGHTS_NORM: 0,
        }

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def configure_optimizers(self):
        return optim.Adam(
            itertools.chain(self.cnn.parameters(), self.cnn.parameters(), self.cnn.parameters(), self.cnn.parameters()),
            lr=self.opt[LR], betas=(self.opt[RUNTIME][BETA1], self.opt[RUNTIME][BETA2]), eps=1e-08)

    def log_img(self, output, img_dirpath, logname, fname):
        imgpath = osp.join(img_dirpath, fname)
        img = saveTensorAsImg(output, imgpath)
        self.logger.experiment.log_image(img, overwrite=False, name=logname)

    def train_forward_one_batch(self, batch):
        '''
        input shape: [b, 3, H, W]
        self.cnn(img) shape: [b, 3, 1, 1]
        '''

        # pred: weights of each LUT
        pred_weights = self.cnn(batch).squeeze()
        if len(pred_weights.shape) == 1:
            # when batchsize is 1:
            pred_weights = pred_weights.unsqueeze(0)

        lut_outputs = [lut(batch) for lut in self.luts]
        weights_norm = torch.mean(pred_weights ** 2)
        combine_A = batch.new(batch.size())

        for b in range(batch.size(0)):
            # for each image in the batch, merge all LUT's outputs with weights
            combine_A[b, :, :, :] = sum([
                pred_weights[b, i] * lut_outputs[i][b, :, :, :]
                for i in range(lut_outputs)
            ])

        return combine_A, weights_norm

    def eval_forward_one_img(self, img):
        '''
        input img shape: [1, 3, H, W]
        '''
        if len(img.shape) != 4:
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            else:
                raise RuntimeWarning(
                    f'WARN: In IA3DLUTLitModel.eval_forward_one_img: input image shape must be [1, 3, H, W], but img.shape={img.shape}')

        pred_weights = self.cnn(img).squeeze()
        final_LUT = sum([
            weight * self.luts[i].LUT
            for i, weight in enumerate(pred_weights)
        ])

        weights_norm = torch.mean(pred_weights ** 2)
        combine_A = img.new(img.size())
        _, combine_A = trilinear_(final_LUT, img)
        return combine_A, weights_norm

    def training_step(self, batch, batch_idx):
        # get output
        input_batch, gt_batch = Variable(batch[INPUT_IMG], requires_grad=False), \
                                Variable(batch[OUTPUT_IMG],
                                         requires_grad=False)
        output_batch, self.train_metrics[WEIGHTS_NORM] = self.train_forward_one_batch(input_batch)

        # calculate loss:
        mse = self.criterion(output_batch, gt_batch)
        tv_mn_pairs = [self.tv3(x) for x in self.luts]
        self.train_metrics[TV_CONS] = sum([x[0] for x in tv_mn_pairs])
        self.train_metrics[MN_CONS] = sum([x[1] for x in tv_mn_pairs])
        loss = (mse +
                self.opt[RUNTIME][LAMBDA_SMOOTH] * (self.train_metrics[WEIGHTS_NORM] +
                                                    self.train_metrics[TV_CONS]) +
                self.opt[RUNTIME][LAMBDA_MONOTONICITY] * self.train_metrics[MN_CONS])

        # get psnr
        self.train_metrics[PSNR] = ImageProcessing.compute_psnr(output_batch, gt_batch, 1.0)

        # log to pl
        for x, y in self.train_metrics.items():
            self.log(x, y)  # , on_step=False, on_epoch=True, prog_bar=True)

        # log to comet
        self.logger.experiment.log_metrics(self.train_metrics)

        # save images
        if batch_idx % self.opt[LOG_EVERY] == 0:
            fname = f'epoch{self.epoch}_iter{batch_idx}.png'
            self.log_img(output_batch, self.train_img_dirpath, OUTPUT_IMG, fname)
        return loss

    def validation_step(self, batch, batch_ix):
        # get output
        input_batch, gt_batch = Variable(batch[INPUT_IMG], requires_grad=False), \
                                Variable(batch[OUTPUT_IMG],
                                         requires_grad=False)
        output_batch, self.valid_metrics[WEIGHTS_NORM] = generator_eval(real_A)

        # save valid images
        if batch_idx % self.opt[LOG_EVERY] == 0:
            fname = f'epoch{self.epoch}_iter{batch_idx}.png'
            self.log_img(output_batch, self.valid_img_dirpath, OUTPUT_IMG, fname)

        # get psnr
        self.valid_metrics[PSNR] = ImageProcessing.compute_psnr(output_batch, gt_batch, 1.0)

        # log to pl and comet
        logged_metrics = {f'{VALID}.{x}': y for x in self.valid_metrics}
        self.logger.experiment.log_metrics(self.logged_metrics)
        for x, y in logged_metrics.items():
            self.log(x, y)

    def training_epoch_end(self, outputs):
        self.epoch += 1

    def test_step(self, batch, batch_ix):
        # TODO 修改这里 这里现在还是从deeplpf复制过来的
        pass
        # # test without GT image:
        # input_batch, fname = batch[INPUT_IMG], batch[NAME]
        # output_dict = self.net(input_batch)
        # output = torch.clamp(output_dict[OUTPUT], 0.0, 1.0)
        #
        # # TODO: 这里的fname是一个单元素list.....为啥啊 太奇怪了
        # fname = fname[0]
        # saveTensorAsImg(output, os.path.join(self.opt[IMG_DIRPATH], osp.basename(fname)))
        # if PREDICT_ILLUMINATION in output_dict:
        #     saveTensorAsImg(
        #         output_dict[PREDICT_ILLUMINATION],
        #         os.path.join(self.illumination_dirpath, osp.basename(fname))
        #     )
        #
        # # test with GT:
        # if OUTPUT_IMG in batch:
        #     # calculate metrics:
        #     output_ = output.clone().detach().cpu().numpy()
        #     y_ = batch[OUTPUT_IMG].clone().detach().cpu().numpy()
        #     psnr = ImageProcessing.compute_psnr(output_, y_, 1.0)
        #     ssim = ImageProcessing.compute_ssim(output_, y_)
        #     for x, y in {PSNR: psnr, SSIM: ssim}.items():
        #         self.log(x, y, on_step=True, on_epoch=True)

    def forward(self, x):
        return self.eval_forward_one_img(x)


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)
        # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        net.fc = nn.Linear(512, out_dim)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        # x = torch._C._nn.upsample_bilinear2d( x, (224,224), align_corners=False )

        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f


##############################
#        Discriminator
##############################


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))

    return layers


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128),
            nn.Conv2d(128, 1, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)


class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Classifier_unpaired(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier_unpaired, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, opt, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        lines = open(opt[RUNTIME][LUT_FILEPATH], 'r').readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        # self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self.LUT = nn.Parameter(self.LUT.clone())
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        assert 1 == trilinear.forward(lut,
                                      x,
                                      output,
                                      dim,
                                      shift,
                                      binsize,
                                      W,
                                      H,
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        assert 1 == trilinear.backward(x,
                                       x_grad,
                                       lut_grad,
                                       dim,
                                       shift,
                                       binsize,
                                       W,
                                       H,
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    # 输入LUT和iamge，输出LUT作用于image的结果。
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        # import pdb; pdb.set_trace()
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))
        return tv, mn


class ImageAdaptive3dLUT(nn.Module):
    def __init__(self):
        self.LUT0 = Generator3DLUT_identity()
        self.LUT1 = Generator3DLUT_zero()
        self.LUT2 = Generator3DLUT_zero()

    def forward(self):
        pass
