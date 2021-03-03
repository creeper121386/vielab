# -*- coding: utf-8 -*-
import math
import os
import os.path as osp

import cv2
# import matplotlib
import numpy as np
import hydra
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from util import ImageProcessing, saveTensorAsImg, checkConfig, configLogging

import model
from data import Adobe5kDataLoader, Dataset
from globalenv import *
import glob


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_com(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_com(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_com(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_com(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def testWithGT(opt, log_dirpath, img_dirpath, net):
    testdata = Dataset(opt, data_dict=None, transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]),
        normaliser=2 ** 8 - 1, is_valid=False)

    dataloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False,
                                             num_workers=4)
    psnr_all = psnr_count = ssim_all = ssim_count = 0
    for batch_num, data in enumerate(dataloader, 0):
        '''
        psnr_count += 1
        if psnr_count <= 201:
            continue
        '''

        x, y, category = Variable(data[INPUT_IMG], requires_grad=False).cuda(), \
            Variable(data[OUTPUT_IMG], requires_grad=False).cuda(), \
            data[NAME]

        path_split = category[0].split('/')
        path_id = path_split[-1].split('.jpg')[0]

        with torch.no_grad():
            outputDict = net(x)
            output = torch.clamp(outputDict[OUTPUT], 0.0, 1.0)

        saveTensorAsImg(output, os.path.join(img_dirpath, path_id + '.jpg'))

        if PREDICT_ILLUMINATION in outputDict:
            # import ipdb; ipdb.set_trace()
            illuminationPath = os.path.join(log_dirpath, PREDICT_ILLUMINATION)
            if not os.path.exists(illuminationPath):
                os.makedirs(illuminationPath)
            saveTensorAsImg(outputDict[PREDICT_ILLUMINATION], os.path.join(
                illuminationPath, path_id + '.jpg'))

        # ─── CALCULATE METRICS ───────────────────────────────────────────
        output_ = output.clone().detach().cpu().numpy()
        y_ = y.clone().detach().cpu().numpy()

        psnr_this = ImageProcessing.compute_psnr(output_, y_, 1.0)
        ssim_this = ImageProcessing.compute_ssim(output_, y_)
        psnr_all += psnr_this
        psnr_count += 1
        ssim_all += ssim_this
        ssim_count += 1

        console.log(
            f'Test [[{psnr_count}]] SSIM: {ssim_this:.4f}, PSNR: {psnr_this:.4f}')
    console.log(f'Average PSNR: {psnr_all * 1.0 / psnr_count}')
    console.log(f'Average SSIM: {ssim_all * 1.0 / ssim_count}')


count = 0


def evalWithoutGT(opt, log_dirpath, img_dirpath, net, path):
    count += 1
    console.log(f'[[Eval {count}]] Now processing {path}')
    x = torch.Tensor(Image.open(path))
    fname = osp.basename(path)

    with torch.no_grad():
        outputDict = net(x)
        output = torch.clamp(outputDict[OUTPUT], 0.0, 1.0)

    saveTensorAsImg(output, os.path.join(img_dirpath, fname))

    if PREDICT_ILLUMINATION in outputDict:
        # import ipdb; ipdb.set_trace()
        illuminationPath = os.path.join(log_dirpath, PREDICT_ILLUMINATION)
        if not os.path.exists(illuminationPath):
            os.makedirs(illuminationPath)
        saveTensorAsImg(outputDict[PREDICT_ILLUMINATION], os.path.join(
            illuminationPath, fname))
    return output.clone().detach().cpu().numpy()


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = checkConfig(opt, TEST)

    checkpoint_filepath = opt[CHECKPOINT_FILEPATH]

    torch.cuda.set_device(opt[GPU])
    console.log('Current cuda device:', torch.cuda.current_device())
    log_dirpath, img_dirpath = configLogging(TEST, opt)

    net = model.DeepLPFNet(opt)
    para = torch.load(checkpoint_filepath,
                      map_location=lambda storage, location: storage)
    # switch model to evaluation mode
    net.load_state_dict(para)
    net.eval()
    net.cuda()

    # ─── LOAD DATA ──────────────────────────────────────────────────────────────────
    if opt[DATA][GT_DIRPATH]:
        testWithGT(opt, log_dirpath, img_dirpath, net)
    else:
        for x in glob.glob(opt[DATA][INPUT_DIRPATH]):
            evalWithoutGT(opt, log_dirpath, img_dirpath, net, x)


if __name__ == "__main__":
    main()
