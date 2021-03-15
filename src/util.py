# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import datetime
import logging
import os
import os.path as osp

import cv2
import numpy as np
import omegaconf
import torch
import yaml
from globalenv import *
from matplotlib.image import imread
from rich import print
from skimage import measure
from torch.autograd import Variable
from torchvision import transforms


# def init_config(opt):
#     # print
#     def hint(k):
#         os.system('clear')
#         console.print(f'[yellow]* Current `{k}` config is:[/yellow]')
#         console.print(opt[k])
#         print()
#
#     opt = omegaconf.OmegaConf.to_container(opt)
#     updated_contens = {}
#
#     # update data, runtime config:
#     for k in [DATA, RUNTIME]:
#         hint(k)
#         console.print('[yellow]* Available values:[/yellow]')
#         value_list = os.listdir(osp.join(CONFIG_DIRPATH, k))
#         completer = FuzzyWordCompleter(value_list)
#         console.print(Markdown(
#                 ''.join(['\n- ' + x for x in value_list])
#             ))
#         print()
#         in_value = None
#         while True:
#             in_value = prompt(f'* Enter the new value of `{k}` (Enter {SKIP_FLAG} to skip): ', completer=completer, complete_while_typing=True)
#             if in_value in [*value_list, SKIP_FLAG]:
#                 break
#
#         if in_value != SKIP_FLAG:
#             opt[k] = parseSingleYmlConfig(osp.join(CONFIG_DIRPATH, k, in_value))
#             updated_contens[k] = opt[k]
#
#     # update path config:
#     for k in [CHECKPOINT_PATH]:
#         path_completer = PathCompleter()
#         hint(k)
#         in_value = None
#         while True:
#             in_value = prompt(f'* Enter the new value of `{k}` (Enter {SKIP_FLAG} to skip): ', completer=path_completer,
#                               complete_while_typing=True)
#             if osp.exists(in_value) or in_value == SKIP_FLAG:
#                 break
#         if in_value != SKIP_FLAG:
#             opt[k] = in_value
#             ted_contens[k] = opt[k]
#
#     # start running
#     os.system('clear')
#     console.log('Start running program.')
#     if len(updated_contens):
#         console.log('You have changed the following config:')
#         console.log(updated_contens)
#         print()
#
#     return opt


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_com(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
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


# (Discarded, now use Hydra)
def parseConfig(dirpath):
    '''
    parse structured configs from directories. The directory must contains `config.yaml` , which defines which configLogging group to choose from. For example:

    config
    ├── config.yaml
    ├── db
    │   ├── mysql.yaml
    │   └── postgresql.yaml
    ├── schema
    │   ├── school.yaml
    │   ├── support.yaml
    │   └── warehouse.yaml
    └── ui
        ├── full.yaml
        └── view.yaml

    And the config.yaml is:

    defaults:
        - db: mysql
        - schema: school
        - ui: view
    '''
    pass


def parseAugmentation(opt):
    '''
    return: pytorch composed transform
    '''
    aug_config = opt[AUGMENTATION]

    aug_list = [transforms.ToPILImage(), ]
    if aug_config[HORIZON_FLIP]:
        aug_list.append(transforms.RandomHorizontalFlip())

    elif aug_config[VERTICAL_FLIP]:
        aug_list.append(transforms.RandomVerticalFlip())

    aug_list.append(transforms.ToTensor())
    return transforms.Compose(aug_list)


def configLogging(mode, opt):
    log_dirpath = f"../{mode}_log/{opt[RUNTIME][MODELNAME]}/{opt[EXPNAME]}_" + \
                  datetime.datetime.now().strftime(LOG_TIME_FORMAT)
    img_dirpath = osp.join(log_dirpath, IMAGES)

    os.makedirs(log_dirpath)
    os.makedirs(img_dirpath)

    console.log('Build log directory:', log_dirpath)
    console.log('Build image directory:', img_dirpath)

    handlers = [logging.FileHandler(osp.join(log_dirpath, LOG_FILENAME)), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    for k, v in opt.items():
        logging.info(f'*** Param - {k}: {v}')

    return log_dirpath, img_dirpath


def saveTensorAsImg(output, path, resize=False):
    outImg = output[0].permute(
        1, 2, 0).clone().detach().cpu().numpy() * 255.0
    outImg = outImg[:, :, [2, 1, 0]].astype(np.uint8)

    if resize:
        assert type(resize + 0.1) == float

        h = outImg.shape[0]
        w = outImg.shape[1]
        outImg = cv2.resize(outImg, (int(w / resize), int(h / resize))).astype(np.uint8)

    cv2.imwrite(path, outImg)
    return outImg


def checkConfig(opt, mode):
    def checkField(opt, name, raise_msg):
        try:
            assert name in opt
        except:
            raise RuntimeError(raise_msg)

    # check necessary argments for ALL MODELS and ALL MODES:
    for x in GENERAL_NECESSARY_ARGUMENTS:
        checkField(opt, x, ARGUMENTS_MISSING_ERRS[x])

    # check necessary argments for all models for EACH MODE:
    if mode == TRAIN:
        necessaryFields = TRAIN_NECESSARY_ARGUMENTS
    elif mode in [TEST, EVAL]:
        necessaryFields = TEST_NECESSARY_ARGUMENTS
    else:
        raise NotImplementedError('ERR: In function [checkConfig]: unknown mode', mode)
    for x in necessaryFields:
        checkField(opt, x, ARGUMENTS_MISSING_ERRS[x])

    # check necessary argments for EACH MODEL:
    assert opt[RUNTIME][MODELNAME] in SUPPORTED_MODELS
    for x in RUNTIME_NECESSARY_ARGUMENTS[opt[RUNTIME][MODELNAME]]:
        checkField(opt[RUNTIME], x, f'ERR: Config missing argument: {x}')

    if type(opt) == omegaconf.DictConfig:
        return omegaconf.OmegaConf.to_container(opt)
    return opt


# (Discarded)
def parseSingleYmlConfig(ymlpath):
    '''
    input config file path (yml file), return config dict.
    '''
    print(f'* Reading config from: {ymlpath}')

    if ymlpath.startswith('http'):
        import requests
        ymlContent = requests.get(ymlpath).content
    else:
        ymlContent = open(ymlpath, 'r').read()

    yml = yaml.load(ymlContent)
    return yml


class ImageProcessing(object):

    @staticmethod
    def rgb_to_lab(img, is_training=True):
        """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: 
        :returns: 
        :rtype: 

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)

        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(
            0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
            [0.412453, 0.212671,
             0.019334],  # R
            [0.357580, 0.715160,
             0.119193],  # G
            [0.180423, 0.072169,
             0.950227],  # B
        ]), requires_grad=False).type_as(img)

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1 / 0.950456, 1.0, 1 / 1.088754]), requires_grad=False).type_as(img))

        epsilon = 6 / 29

        img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + \
              (torch.clamp(img, min=0.0001) ** (1.0 / 3.0) * img.gt(epsilon ** 3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0, 500.0, 0.0],  # fx
                                                    [116.0, -500.0, 200.0],  # fy
                                                    [0.0, 0.0, -200.0],  # fz
                                                    ]), requires_grad=False).type_as(img)

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).type_as(img)

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img[0, :, :] = img[0, :, :] / 100
        img[1, :, :] = (img[1, :, :] / 110 + 1) / 2
        img[2, :, :] = (img[2, :, :] / 110 + 1) / 2

        img[(img != img).detach()] = 0

        img = img.contiguous()

        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        """Move the image channels to the last dimensiion of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        img = ImageProcessing.normalise_image(
            imread(img_filepath), normaliser)  # NB: imread normalises to 0-1
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                        np.log10(max_intensity ** 2 /
                                 ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        """Computes the SSIM for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        ssim_val = 0.0

        for i in range(0, num_images):
            imageA = ImageProcessing.swapimdims_3HW_HW3(
                image_batchA[i, 0:3, :, :])
            imageB = ImageProcessing.swapimdims_3HW_HW3(
                image_batchB[i, 0:3, :, :])
            ssim_val += measure.compare_ssim(imageA, imageB, data_range=imageA.max() - imageA.min(), multichannel=True,
                                             gaussian_weights=True, win_size=11)

        return ssim_val / num_images
