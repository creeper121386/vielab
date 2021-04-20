import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from .basic_loss import L_TV, L_spa, L_color, L_exp
from globalenv import *
from torchvision.models.vgg import vgg16

from .basemodel import BaseModel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ZeroDCELitModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt, [TRAIN])
        if opt[VALID_DATA][INPUT]:
            console.log('[yellow]WARNING: valid mode is not needed in ZeroDCE, ignore valid_dataloader.[/yellow]')

        self.net = enhance_net_nopool(opt)
        self.net.apply(weights_init)

        self.color_loss = L_color()
        self.spatial_loss = L_spa()
        self.exposure_loss = L_exp(16, 0.6)
        self.tvloss = L_TV()

        self.net.train()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.opt[LR], weight_decay=self.opt[RUNTIME][WEIGHT_DECAY])
        return optimizer

    def training_step(self, batch, batch_idx):
        input_batch = batch[INPUT]
        enhanced_image_1, enhanced_image, A = self.net(input_batch)

        loss_tv = 200 * self.tvloss(A)
        loss_spatial = torch.mean(self.spatial_loss(enhanced_image, input_batch))
        loss_color = 5 * torch.mean(self.color_loss(enhanced_image))
        loss_exposure = 10 * torch.mean(self.exposure_loss(enhanced_image))
        loss = loss_tv + loss_spatial + loss_color + loss_exposure

        self.log_dict({
            LTV_LOSS: loss_tv,
            SPATIAL_LOSS: loss_spatial,
            COLOR_LOSS: loss_color,
            EXPOSURE_LOSS: loss_exposure,
            LOSS: loss
        })

        # logging image:
        if self.global_step % self.opt[LOG_EVERY] == 0:
            fname = osp.basename(batch[FPATH][0]) + f'_epoch{self.current_epoch}_iter{self.global_step}.png'
            self.save_one_img_of_batch(enhanced_image, self.train_img_dirpath, fname)
            self.add_img_to_buffer(TRAIN, input_batch, TRAIN, INPUT, fname)
            self.add_img_to_buffer(TRAIN, enhanced_image, TRAIN, OUTPUT, fname)
            self.commit_logger_buffer(TRAIN)
        return loss

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm(self.net.parameters(), self.opt[RUNTIME][GRAD_CLIP_NORM])

    def test_step(self, batch, batch_ix):
        input_batch, fname = batch[INPUT], batch[FPATH][0]
        _, output_batch, _ = self.net(input_batch)

        # batchsize must be 1:
        assert output_batch.shape[0] == 1

        dst_fname = os.path.join(self.opt[IMG_DIRPATH], osp.basename(fname))
        torchvision.utils.save_image(output_batch[0], dst_fname)

    def forward(self, x):
        _, enhanced_image, _ = self.net(x)
        return enhanced_image


### MODEL AND LOSS DEFINATION:


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


class enhance_net_nopool(nn.Module):

    def __init__(self, opt):
        super(enhance_net_nopool, self).__init__()
        self.opt = opt
        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)

        if self.opt[RUNTIME][PREDICT_ILLUMINATION]:
            self.e_conv7 = nn.Conv2d(number_f * 2, 3, 3, 1, 1, bias=True)
        else:
            self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.opt[RUNTIME][PREDICT_ILLUMINATION]:
            # predict illumination map:
            output = x / (torch.where(x_r < x, x, x_r) + 1e-7)
            # output = x * x_r
            return None, output, x_r

        else:
            # predict 8 alpha maps:
            r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
            x = x + r1 * (torch.pow(x, 2) - x)
            x = x + r2 * (torch.pow(x, 2) - x)
            x = x + r3 * (torch.pow(x, 2) - x)
            enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
            x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
            x = x + r6 * (torch.pow(x, 2) - x)
            x = x + r7 * (torch.pow(x, 2) - x)
            enhance_image = x + r8 * (torch.pow(x, 2) - x)
            r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
            return enhance_image_1, enhance_image, r
