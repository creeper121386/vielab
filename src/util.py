# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
from torch.autograd import Variable
from skimage import measure

# from skimage.measure import compare_ssim as ssim
from matplotlib.image import imread, imsave
import torch
import numpy as np
import os
import os.path as osp
import cv2
# import matplotlib
import yaml
import logging
import datetime
from globalenv import *

# matplotlib.use('agg')


def configLogging(mode, opt):
    log_dirpath = f"../{mode}_log/{opt[EXPNAME]}_" + \
        datetime.datetime.now().strftime(TIME_FORMAT)
    img_dirpath = osp.join(log_dirpath, IMAGES)

    os.makedirs(log_dirpath)
    os.makedirs(img_dirpath)

    console.log('Build log directory:', log_dirpath)
    console.log('Build image directory:', img_dirpath)

    handlers = [logging.FileHandler(
        log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    for k, v in opt.items():
        logging.info(f'### Param - {k}: {v}')

    return log_dirpath, img_dirpath


def saveTensorAsImg(output, path, resize=False):
    outImg = output[0].permute(
        1, 2, 0).clone().detach().cpu().numpy() * 255.0
    outImg = outImg[:, :, [2, 1, 0]].astype(np.uint8)

    if resize:
        assert type(resize + 0.1) == float

        h = outImg.shape[0]
        w = outImg.shape[1]
        outImg = cv2.resize(outImg, (int(w / resize), int(h / resize)))

    cv2.imwrite(path, outImg.astype(np.uint8))


def parseConfig(ymlpath, mode):
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
    if mode == TRAIN:
        necessaryFields = trainNecessaryFields
    elif mode == TEST:
        necessaryFields = testNecessaryFields
    else:
        raise NotImplementedError('Function[parseConfig]: unknown mode', mode)

    # make sure the format is valid:
    for x in necessaryFields:
        try:
            assert x in yml
        except:
            print('Field missing:', x)

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
                                                                       min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
                                                [0.412453, 0.212671,
                                                    0.019334],  # R
                                                [0.357580, 0.715160,
                                                    0.119193],  # G
                                                [0.180423, 0.072169,
                                                    0.950227],  # B
                                                ]), requires_grad=False).cuda()

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1/0.950456, 1.0, 1/1.088754]), requires_grad=False).cuda())

        epsilon = 6/29

        img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
            (torch.clamp(img, min=0.0001)**(1.0/3.0) * img.gt(epsilon**3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0,  500.0,    0.0],  # fx
                                                    [116.0, -500.0,  200.0],  # fy
                                                    [0.0,    0.0, -200.0],  # fz
                                                    ]), requires_grad=False).cuda()

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).cuda()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img[0, :, :] = img[0, :, :]/100
        img[1, :, :] = (img[1, :, :]/110 + 1)/2
        img[2, :, :] = (img[2, :, :]/110 + 1)/2

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