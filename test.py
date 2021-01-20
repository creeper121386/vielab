# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

Instructions:

To get this code working on your system / problem you will need to edit the
data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data
directory (anything with "/aiml/data")

2. data.py, lines 216, 224, change the folder names of the data input and
output directories to point to your folder names
'''
import model
import metric
import os
import glob
from skimage.measure import compare_ssim as ssim
import os.path
import torch.nn.functional as F
from math import exp
from skimage import io, color
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from scipy.ndimage.filters import convolve
import torch.nn.init as net_init
import datetime
from util import ImageProcessing
import math
import numpy as np
import copy
import torch.optim as optim
import shutil
import argparse
from shutil import copyfile
from PIL import Image
import logging
# import data
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import torchvision.transforms as transforms
import traceback
import torch.nn as nn
import torch
import time
import random
import skimage
import unet
from data_rx import Adobe5kDataLoader, Dataset
from abc import ABCMeta, abstractmethod
import imageio
import cv2
from skimage.transform import resize
import matplotlib
matplotlib.use('agg')
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

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "./log_" + timestamp
    os.mkdir(log_dirpath)

    handlers = [logging.FileHandler(log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False, help="Number of epoches (default 5000)", default=100000)
    parser.add_argument(
        "--valid_every", type=int, required=False, help="Number of epoches after which to compute validation accuracy",
        default=500)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved DeepLPF model instance", default=None)

    args = parser.parse_args()
    num_epoch = args.num_epoch
    valid_every = args.valid_every
    checkpoint_filepath = args.checkpoint_filepath
    inference_img_dirpath = args.inference_img_dirpath

    logging.info('######### Parameters #########')
    logging.info('Number of epochs: ' + str(num_epoch))
    logging.info('Logging directory: ' + str(log_dirpath))
    logging.info('Dump validation accuracy every: ' + str(valid_every))
    logging.info('##############################')


    '''
    training_data_loader = Adobe5kDataLoader(data_dirpath="/aiml/data/",
                                             img_ids_filepath="/aiml/data/images_train.txt")
    training_data_dict = training_data_loader.load_data()
    '''

    training_dataset = Dataset(data_dict=None, transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]),
        normaliser=2 ** 8 - 1, is_valid=False)


    '''
    validation_data_loader = Adobe5kDataLoader(data_dirpath="/aiml/data/",
                                               img_ids_filepath="/aiml/data/images_valid.txt")
    validation_data_dict = validation_data_loader.load_data()
    validation_dataset = Dataset(data_dict=validation_data_dict,
                                 transform=transforms.Compose([transforms.ToTensor()]), normaliser=2 ** 8 - 1,
                                 is_valid=True)

    testing_data_loader = Adobe5kDataLoader(data_dirpath="/aiml/data/",
                                            img_ids_filepath="/aiml/data/images_test.txt")
    testing_data_dict = testing_data_loader.load_data()
    testing_dataset = Dataset(data_dict=testing_data_dict,
                              transform=transforms.Compose([transforms.ToTensor()]), normaliser=2 ** 8 - 1,
                              is_valid=True)
    '''
    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=False,
                                                       num_workers=4)

    '''
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False,
                                                      num_workers=4)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1,
                                                         shuffle=False,
                                                         num_workers=4)
    '''

    net = model.DeepLPFNet()
    # checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-11-15_00-54-57/deep_lpf_20.pth'

    # checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-23_20-34-41/deep_lpf_180.pth'
    # checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-24_13-12-53/deep_lpf_180.pth'
    # checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-24_23-27-40/deep_lpf_180.pth'
    # checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-24_23-32-08/deep_lpf_140.pth'
    # checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-24_23-34-28/deep_lpf_361.pth'

    # checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-24_23-27-40/deep_lpf_200.pth'
    checkpoint_filepath = '/mnt/proj3/xgxu/comparison/DeepLPF/log_2020-12-25_17-57-26/deep_lpf_400.pth'
    para = torch.load(checkpoint_filepath, map_location=lambda storage, location: storage)
    # switch model to evaluation mode
    net.load_state_dict(para)
    net.eval()

    logging.info('######### Network created #########')
    logging.info('Architecture:\n' + str(net))

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    # criterion = model.DeepLPFLoss(ssim_window_size=5)

    '''
    The following objects allow for evaluation of a model on the testing and validation splits of a dataset
    '''
    '''
    validation_evaluator = metric.Evaluator(
        criterion, validation_data_loader, "valid", log_dirpath)
    testing_evaluator = metric.Evaluator(
        criterion, testing_data_loader, "test", log_dirpath)
    '''
    print('1111')
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    # best_valid_psnr = 0.0
    # alpha = 0.0
    # optimizer.zero_grad()
    # net.train()

    running_loss = 0.0
    examples = 0
    psnr_avg = 0.0
    ssim_avg = 0.0
    batch_size = 1
    net.cuda()

    count = 0
    psnr_all = 0
    psnr_count = 0
    ssim_all = 0
    ssim_count = 0
    for batch_num, data in enumerate(training_data_loader, 0):
        '''
        psnr_count += 1
        if psnr_count <= 201:
            continue
        '''
        input_img_batch, output_img_batch, category = Variable(data['input_img'], requires_grad=False).cuda(), \
                                                      Variable(data['output_img'], requires_grad=False).cuda(), \
                                                      data['name']

        path_split = category[0].split('/')
        # path_id = path_split[-2] + '_' + path_split[-1]
        path_id = path_split[-1].split('.jpg')[0]

        start_time = time.time()
        with torch.no_grad():
            net_output_img_batch = net(input_img_batch)
            net_output_img_batch = torch.clamp(net_output_img_batch, 0.0, 1.0)

        # print(input_img_batch.shape, net_output_img_batch.shape, torch.max(input_img_batch), torch.min(input_img_batch))

        output_image11 = net_output_img_batch[0, :, :, :]
        output_image11 = output_image11.permute(1, 2, 0)
        output_image11 = output_image11.clone().detach().cpu().numpy()
        output_image11 = output_image11 * 255.0
        output_image11 = output_image11[:, :, [2, 1, 0]].astype(np.uint8)
        result_dir = "new_results/5_jpg_under"
        if not(os.path.exists(result_dir)):
            os.mkdir(result_dir)

        h = output_image11.shape[0]
        w = output_image11.shape[1]
        output_image11 = cv2.resize(output_image11, (w*3//2, h*3//2))

        cv2.imwrite(os.path.join(result_dir, path_id + '.jpg'), output_image11.astype(np.uint8))


        '''
        output_image12 = output_img_batch[0, :, :, :]
        output_image12 = output_image12.permute(1, 2, 0)
        output_image12 = output_image12.clone().detach().cpu().numpy()
        output_image12 = output_image12 * 255.0
        output_image12 = output_image12[:, :, [2, 1, 0]].astype(np.uint8)
        psnr_this = calculate_psnr(output_image11, output_image12)
        ssim_this = calculate_ssim(output_image11, output_image12)
        '''
        psnr_this = 0
        ssim_this = 0
        psnr_all += psnr_this
        psnr_count += 1
        ssim_all += ssim_this
        ssim_count += 1

        print(ssim_this, psnr_this, psnr_count)
    print(psnr_all * 1.0 / psnr_count)
    print(ssim_all * 1.0 / ssim_count)


if __name__ == "__main__":
    main()
