# -*- coding: utf-8 -*-
from rich.console import Console
import argparse
import datetime
import logging
import math
import os
import os.path
import time


import cv2
import matplotlib
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from util import ImageProcessing

import model
from data_rx import Adobe5kDataLoader, Dataset

matplotlib.use('agg')
console = Console()
# np.set_printoptions(threshold=np.nan)


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


def main():
    # ─── LOGGING ────────────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    log_dirpath = "./test_log/" + timestamp
    os.makedirs(log_dirpath)

    handlers = [logging.FileHandler(
        log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    # ─── PARSE CONFIG ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")
    parser.add_argument(
        "--checkpoint_filepath", '-m', required=True, help="Location of checkpoint file")
    parser.add_argument(
        "--GT_dirpath", '-g', required=True, help="GT dir path")
    parser.add_argument(
        "--input_dirpath", '-i', required=True, help="input dir path")

    args = parser.parse_args()
    checkpoint_filepath = args.checkpoint_filepath
    opt = args.__dict__

    # ─── LOAD DATA ──────────────────────────────────────────────────────────────────
    d = Dataset(opt, data_dict=None, transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]),
        normaliser=2 ** 8 - 1, is_valid=False)

    dataloader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False,
                                                       num_workers=4)

    net = model.DeepLPFNet()
    para = torch.load(checkpoint_filepath,
                      map_location=lambda storage, location: storage)
    # switch model to evaluation mode
    net.load_state_dict(para)
    net.eval()

    # ─── PRINT NET LAYERS ───────────────────────────────────────────────────────────
    # logging.info('######### Network created #########')
    # logging.info('Architecture:\n' + str(net))
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # ───  ───────────────────────────────────────────────────────────────────────────

    net.cuda()

    count = 0
    psnr_all = 0
    psnr_count = 0
    ssim_all = 0
    ssim_count = 0
    for batch_num, data in enumerate(dataloader, 0):
        '''
        psnr_count += 1
        if psnr_count <= 201:
            continue
        '''
        x, y, category = Variable(data['input_img'], requires_grad=False).cuda(), \
            Variable(data['output_img'], requires_grad=False).cuda(), \
            data['name']

        path_split = category[0].split('/')
        # path_id = path_split[-2] + '_' + path_split[-1]
        path_id = path_split[-1].split('.jpg')[0]

        start_time = time.time()
        with torch.no_grad():
            output = net(x)
            output = torch.clamp(output, 0.0, 1.0)

        outImg = output[0].permute(
            1, 2, 0).clone().detach().cpu().numpy() * 255.0
        outImg = outImg[:, :, [2, 1, 0]].astype(np.uint8)
        result_dir = "results/"
        if not(os.path.exists(result_dir)):
            os.mkdir(result_dir)

        h = outImg.shape[0]
        w = outImg.shape[1]
        outImg = cv2.resize(outImg, (w*3//2, h*3//2))

        cv2.imwrite(os.path.join(result_dir, path_id + '.jpg'),
                    outImg.astype(np.uint8))

        # import ipdb; ipdb.set_trace()
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


if __name__ == "__main__":
    main()
