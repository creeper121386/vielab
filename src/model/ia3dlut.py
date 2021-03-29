import itertools
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import trilinear
except:
    Warning('WARN: Can not import module `trilinear`')
import util
from globalenv import *
from torch.autograd import Variable
from .basemodel import BaseModel


class IA3DLUTLitModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.luts = torch.nn.ModuleList([
            Generator3DLUT_identity(opt),
            Generator3DLUT_zero(),
            Generator3DLUT_zero()
        ])
        self.cnn = Classifier()
        self.tv3 = TV_3D()
        self.trilinear = TrilinearInterpolation()

        console.log('Running initialization for IA3DLUTLitModel')
        if not opt[CHECKPOINT_PATH]:
            # evne if when passing opt[CHECKPOINT_PATH], the LitModel will execute the __init__ method.
            # so when passing CHECKPOINT_PATH, skip classifier parameter init.
            self.cnn.apply(weights_init_normal_classifier)
            torch.nn.init.constant_(self.cnn.model[16].bias.data, 1.0)

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

    def configure_optimizers(self):

        return optim.Adam(
            itertools.chain(self.cnn.parameters(), *[x.parameters() for x in self.luts]),
            lr=self.opt[LR], betas=(self.opt[RUNTIME][BETA1], self.opt[RUNTIME][BETA2]), eps=1e-08)

    def train_forward_one_batch(self, batch):
        '''
        input shape: [b, 3, H, W]
        self.cnn(img) shape: [b, 3, 1, 1]
        '''

        # pred: weights of each LUT
        for x in self.luts:
            x.to_device(batch.device)
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
                for i in range(len(lut_outputs))
            ])

        return combine_A, weights_norm

    def eval_forward_one_img(self, img):
        '''
        input img shape: [1, 3, H, W]
        '''
        for x in self.luts:
            x.to_device(img.device)

        if len(img.shape) != 4:
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            else:
                raise RuntimeWarning(
                    f'WARN: In IA3DLUTLitModel.eval_forward_one_img: input image shape must be [1, 3, H, W], but img.shape={img.shape}')
        elif img.shape[1] != 1:
            img = img[0].unsqueeze(0)

        pred_weights = self.cnn(img).squeeze()
        # import ipdb; ipdb.set_trace()

        # lut device: cpu
        final_LUT = sum([
            weight * self.luts[i].LUT.type_as(weight)
            for i, weight in enumerate(pred_weights)
        ])

        weights_norm = torch.mean(pred_weights ** 2)
        combine_A = img.new(img.size())
        _, combine_A = self.trilinear(final_LUT, img)
        return combine_A, weights_norm

    def training_step(self, batch, batch_idx):
        # get output
        input_batch, gt_batch = Variable(batch[INPUT_IMG], requires_grad=False), \
                                Variable(batch[OUTPUT_IMG], requires_grad=False)
        output_batch, self.train_metrics[WEIGHTS_NORM] = self.train_forward_one_batch(input_batch)

        # calculate loss:
        mse = self.criterion(output_batch, gt_batch)
        tv_mn_pairs = [self.tv3(x) for x in self.luts]
        self.train_metrics[TV_CONS] = sum([x[0] for x in tv_mn_pairs])
        self.train_metrics[MN_CONS] = sum([x[1] for x in tv_mn_pairs])

        # TODO: 多卡训练这里也有问题：
        loss = (mse +
                self.opt[RUNTIME][LAMBDA_SMOOTH] * (self.train_metrics[WEIGHTS_NORM] + self.train_metrics[TV_CONS]) +
                self.opt[RUNTIME][LAMBDA_MONOTONICITY] * self.train_metrics[MN_CONS])

        self.train_metrics[MSE] = mse
        self.train_metrics[LOSS] = loss

        # get psnr
        self.train_metrics[PSNR] = util.ImageProcessing.compute_psnr(
            util.cuda_tensor_to_ndarray(output_batch),
            util.cuda_tensor_to_ndarray(gt_batch), 1.0
        )

        # log to logger
        for x, y in self.train_metrics.items():
            self.log(x, y, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # get output
        input_batch, gt_batch, fname = Variable(batch[INPUT_IMG], requires_grad=False), \
                                       Variable(batch[OUTPUT_IMG], requires_grad=False), batch[FNAME]
        if type(fname) == list:
            fname = fname[0]
        output_batch, self.valid_metrics[WEIGHTS_NORM] = self.eval_forward_one_img(input_batch)

        # save valid images
        if batch_idx % self.opt[LOG_EVERY] == 0:
            fname = osp.basename(fname) + f'_epoch{self.epoch}_iter{batch_idx}.png'
            self.log_img(output_batch, self.valid_img_dirpath, fname)
            self.add_img_to_logger_buffer(output_batch, OUTPUT_IMG, fname)
            self.add_img_to_logger_buffer(input_batch, INPUT_IMG, fname)
            self.add_img_to_logger_buffer(gt_batch, 'GT', fname)
            self.commit_logger_buffer()

        # get psnr
        self.valid_metrics[PSNR] = util.ImageProcessing.compute_psnr(
            util.cuda_tensor_to_ndarray(output_batch),
            util.cuda_tensor_to_ndarray(gt_batch), 1.0
        )

        # log to pl and logger
        valid_metrics = {f'{VALID}.{x}': y for x, y in self.valid_metrics.items()}

        for x, y in valid_metrics.items():
            # Tips: if call self.log with on_step=True here, metrics will bacome "valid.psnr/epoch_xxx"
            # So just call without arguments.
            self.log(x, y, prog_bar=True)

        return output_batch

    def test_step(self, batch, batch_ix):
        # test without GT image:
        input_batch, fname = batch[INPUT_IMG], batch[FNAME]
        output, _ = self.eval_forward_one_img(input_batch)

        # TODO: 这里的fname是一个单元素list.....为啥啊 太奇怪了
        if type(fname) == list:
            fname = fname[0]
        util.saveTensorAsImg(output, osp.join(self.opt[IMG_DIRPATH], osp.basename(fname)))

        # test with GT:
        if OUTPUT_IMG in batch:
            # calculate metrics:
            psnr = util.ImageProcessing.compute_psnr(
                util.cuda_tensor_to_ndarray(output),
                util.cuda_tensor_to_ndarray(batch[OUTPUT_IMG]), 1.0
            )
            self.log(PSNR, psnr, prog_bar=True)
        return output

    def forward(self, x):
        return self.eval_forward_one_img(x)


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


#
# class resnet18_224(nn.Module):
#
#     def __init__(self, out_dim=5, aug_test=False):
#         super(resnet18_224, self).__init__()
#
#         self.aug_test = aug_test
#         net = models.resnet18(pretrained=True)
#         # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
#         # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
#
#         self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
#
#         net.fc = nn.Linear(512, out_dim)
#         self.model = net
#
#     def forward(self, x):
#         x = self.upsample(x)
#         # x = torch._C._nn.upsample_bilinear2d( x, (224,224), align_corners=False )
#
#         if self.aug_test:
#             # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
#             x = torch.cat((x, torch.flip(x, [3])), 0)
#         f = self.model(x)
#
#         return f


##############################
#        Discriminator
##############################

#
def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))

    return layers


#
#
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=3):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
#             nn.Conv2d(3, 16, 3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.InstanceNorm2d(16, affine=True),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 128),
#             # *discriminator_block(128, 128),
#             nn.Conv2d(128, 1, 8, padding=0)
#         )
#
#     def forward(self, img_input):
#         return self.model(img_input)


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


#
#
# class Classifier_unpaired(nn.Module):
#     def __init__(self, in_channels=3):
#         super(Classifier_unpaired, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
#             nn.Conv2d(3, 16, 3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.InstanceNorm2d(16, affine=True),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 128),
#             # *discriminator_block(128, 128),
#             nn.Conv2d(128, 3, 8, padding=0),
#         )
#
#     def forward(self, img_input):
#         return self.model(img_input)


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

    def to_device(self, device):
        if self.LUT.device != device:
            self.LUT.to(device)

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

    def to_device(self, device):
        if self.LUT.device != device:
            self.LUT.to(device)


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()
        lut = lut.type_as(x)

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

        # console.log(x)
        # import ipdb; ipdb.set_trace()
        int_package = torch.IntTensor([dim, shift, W, H, batch]).type_as(x)
        float_package = torch.FloatTensor([binsize]).type_as(x)
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        # import ipdb; ipdb.set_trace()
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

        # so strange here !!!!!!!!!!
        # lut_grad = lut_grad.cpu()
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    # 输入LUT和iamge，输出LUT作用于image的结果。
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
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

        if self.weight_b.device != dif_r:
            self.weight_r = self.weight_r.type_as(dif_r)
            self.weight_g = self.weight_g.type_as(dif_r)
            self.weight_b = self.weight_b.type_as(dif_r)

        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))
        return tv, mn
