# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import os
from skimage.measure import compare_ssim as ssim
import os.path
import torch.nn.functional as F
from skimage import io, color
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from scipy.ndimage.filters import convolve
import torch.nn.init as net_init
import datetime
import util
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
from collections import defaultdict
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
from abc import ABCMeta, abstractmethod
import imageio
import cv2
from skimage.transform import resize
import matplotlib
matplotlib.use('agg')
# np.set_printoptions(threshold=np.nan)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, transform=None, normaliser=2 ** 8 - 1, is_valid=False):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param transform: PyTorch image transformations to apply to the images
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.data_dict = data_dict
        self.normaliser = normaliser
        self.is_valid = is_valid


        '''
        data_dir_GT = '/mnt/proj3/xgxu/EDVR/datasets/indoor_resize/GT'
        data_dir_input = '/mnt/proj3/xgxu/EDVR/datasets/indoor_resize/input'
        # testing_dir = 'pair13,pair21,pair23,pair31,pair33,pair15'
        testing_dir = 'pair13,pair31'
        # testing_dir = 'pair21,pair23,pair15,pair33'
        # testing_dir = 'pair21'
        '''

        # log_2020-12-24_10-34-14_overexpose
        # dir_this_GT = '/mnt/proj23/rxwang/laiqu_overexpose_correct/overexpose_gt'
        # dir_this_input = '/mnt/proj23/rxwang/laiqu_overexpose_correct/overexpose_input'

        # log_2020-12-24_13-12-53
        # dir_this_GT = '/mnt/proj23/rxwang/underexpose/gt'
        # dir_this_input = '/mnt/proj23/rxwang/underexpose/input'

        # log_2020-12-24_23-27-40
        # dir_this_GT = '/mnt/proj23/rxwang/laiqu_outdoor_data/training/overexpose_gt'
        # dir_this_input = '/mnt/proj23/rxwang/laiqu_outdoor_data/training/overexpose_input'

        # log_2020-12-25_17-57-26
        # dir_this_GT = '/mnt/proj23/rxwang/laiqu_outdoor_data/training/underexpose_gt'
        # dir_this_input = '/mnt/proj23/rxwang/laiqu_outdoor_data/training/underexpose_input'

        # log_2020-12-25_18-01-28
        # dir_this_GT = '/mnt/proj23/rxwang/laiqu_indoor_data/gt_indoor'
        # dir_this_input = '/mnt/proj23/rxwang/laiqu_indoor_data/input_indoor'

        # log_2020-12-25_15-42-42
        ## dir_this_GT = '/mnt/proj23/rxwang/new_UPEnet/outdoor_pair_downsample/gt'
        ## dir_this_input = '/mnt/proj23/rxwang/new_UPEnet/outdoor_pair_downsample/input'



        ## test
        # dir_this_GT = '/mnt/proj23/rxwang/pg_results/train/overexpose_input'
        # dir_this_input = '/mnt/proj23/rxwang/pg_results/train/overexpose_input'
        # dir_this_GT = '/mnt/proj23/rxwang/pg_results/val/under'
        # dir_this_input = '/mnt/proj23/rxwang/pg_results/val/under'
        # dir_this_GT = '/mnt/proj23/rxwang/laiqu_indoor_data/gt_indoor'
        # dir_this_input = '/mnt/proj23/rxwang/laiqu_indoor_data/input_indoor'

        # dir_this_GT = '/mnt/proj23/rxwang/testing_1/9_jpg'
        # dir_this_input = '/mnt/proj23/rxwang/testing_1/9_jpg'

        dir_this_GT = '/mnt/proj23/rxwang/classification/5_jpg_under'
        dir_this_input = '/mnt/proj23/rxwang/classification/5_jpg_under'

        self.train_ids = []

        name_list_GT = []
        name_list_input = []
        for im_name in os.listdir(dir_this_GT):
            name_list_GT.append(im_name)
        for im_name in os.listdir(dir_this_input):
            name_list_input.append(im_name)
        name_list_GT.sort()
        name_list_input.sort()
        assert len(name_list_GT) == len(name_list_input)
        print(len(name_list_GT), 'length')
        for nn in range(len(name_list_GT)):
            self.train_ids.append([os.path.join(dir_this_GT, name_list_GT[nn]), os.path.join(dir_this_input, name_list_input[nn])])

    def __len__(self):
        """Returns the number of images in the dataset

        :returns: number of images in the dataset
        :rtype: Integer

        """
        return (len(self.train_ids))

    def __getitem__(self, idx):
        """Returns a pair of images with the given identifier. This is lazy loading
        of data into memory. Only those image pairs needed for the current batch
        are loaded.

        :param idx: image pair identifier
        :returns: dictionary containing input and output images and their identifier
        :rtype: dictionary

        """


        '''
        output_img = util.ImageProcessing.load_image(
            self.train_ids[idx][0], normaliser=1)
        input_img = util.ImageProcessing.load_image(
            self.train_ids[idx][1], normaliser=1)
        '''


        output_img = cv2.imread(self.train_ids[idx][0])[:, :, [2,1,0]]
        input_img = cv2.imread(self.train_ids[idx][1])[:, :, [2, 1, 0]]


        height = output_img.shape[0]
        width = output_img.shape[1]
        # print(height, width)
        output_img = cv2.resize(output_img, (width*2//3, height*2//3))
        input_img = cv2.resize(input_img, (width*2//3, height*2//3))



        '''
        height_this = 960
        width_this = 960
        rnd_h = random.randint(0, max(0, height-height_this))
        rnd_w = random.randint(0, max(0, width-width_this))
        output_img = output_img[rnd_h:rnd_h + height_this, rnd_w:rnd_w + width_this, :]
        input_img = input_img[rnd_h:rnd_h + height_this, rnd_w:rnd_w + width_this, :]
        '''

        # print(np.mean(output_img))
        input_img = input_img.astype(np.uint8)
        output_img = output_img.astype(np.uint8)

        '''
        if output_img.shape[2] == 4:
            output_img = np.delete(output_img, 3, -1)
        if input_img.shape[2] == 4:
            input_img = np.delete(input_img, 3, -1)
        '''

        seed = random.randint(0, 100000)
        random.seed(seed)
        input_img = self.transform(input_img)
        random.seed(seed)
        output_img = self.transform(output_img)

        return {'input_img': input_img, 'output_img': output_img,
                'name': self.train_ids[idx][1]}


class DataLoader():

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        self.data_dirpath = data_dirpath
        self.img_ids_filepath = img_ids_filepath

    @abstractmethod
    def load_data(self):
        """Abstract function for the data loader class

        :returns: N/A
        :rtype: N/A

        """
        pass

    @abstractmethod
    def perform_inference(self, net, data_dirpath):
        """Abstract function for the data loader class

        :returns: N/A
        :rtype: N/A

        """
        pass


class Adobe5kDataLoader(DataLoader):

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        super().__init__(data_dirpath, img_ids_filepath)
        self.data_dict = defaultdict(dict)

    def load_data(self):
        """ Loads the Samsung image data into a Python dictionary

        :returns: Python two-level dictionary containing the images
        :rtype: Dictionary of dictionaries

        """

        logging.info("Loading Adobe5k dataset ...")

        with open(self.img_ids_filepath) as f:
            '''
            Load the image ids into a list data structure
            '''
            image_ids = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            image_ids_list = [x.rstrip() for x in image_ids]

        idx = 0
        idx_tmp = 0
        img_id_to_idx_dict = {}

        for root, dirs, files in os.walk(self.data_dirpath):

            for file in files:

                img_id = file.split(".")[0]

                is_id_in_list = False
                for img_id_test in image_ids_list:
                    if img_id_test == img_id:
                        is_id_in_list = True
                        break

                if is_id_in_list:  # check that the image is a member of the appropriate training/test/validation split

                    if not img_id in img_id_to_idx_dict.keys():
                        img_id_to_idx_dict[img_id] = idx
                        self.data_dict[idx] = {}
                        self.data_dict[idx]['input_img'] = None
                        self.data_dict[idx]['output_img'] = None
                        idx_tmp = idx
                        idx += 1
                    else:
                        idx_tmp = img_id_to_idx_dict[img_id]

                    if "input" in root:  # change this to the name of your
                                        # input data folder

                        input_img_filepath = file

                        self.data_dict[idx_tmp]['input_img'] = root + \
                            "/" + input_img_filepath

                    elif ("output" in root):  # change this to the name of your
                                             # output data folder

                        output_img_filepath = file

                        self.data_dict[idx_tmp]['output_img'] = root + \
                            "/" + output_img_filepath

                else:

                    logging.debug("Excluding file with id: " + str(img_id))

        for idx, imgs in self.data_dict.items():
            assert ('input_img' in imgs)
            assert ('output_img' in imgs)

        return self.data_dict
