# -*- coding: utf-8 -*-
import logging
import os
import os.path
import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use('agg')

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

        dir_this_GT = '/data1/why/deep_local_parametric_filters/data.down2/input'
        dir_this_input = '/data1/why/deep_local_parametric_filters/data.down2/output'

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
        print('Dataset length: ', len(name_list_GT))
        for nn in range(len(name_list_GT)):
            self.train_ids.append([os.path.join(dir_this_GT, name_list_GT[nn]), os.path.join(
                dir_this_input, name_list_input[nn])])

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

        output_img = cv2.imread(self.train_ids[idx][0])[:, :, [2, 1, 0]]
        input_img = cv2.imread(self.train_ids[idx][1])[:, :, [2, 1, 0]]

        height = output_img.shape[0]
        width = output_img.shape[1]
        # print(height, width)
        output_img = cv2.resize(output_img, (width*2//3, height*2//3))
        input_img = cv2.resize(input_img, (width*2//3, height*2//3))

        # print(np.mean(output_img))
        input_img = input_img.astype(np.uint8)
        output_img = output_img.astype(np.uint8)

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
