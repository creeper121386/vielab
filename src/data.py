# -*- coding: utf-8 -*-
import logging
import os
import os.path
import random
from abc import abstractmethod
from collections import defaultdict

import cv2
# import matplotlib
import numpy as np
import torch
from globalenv import *

# matplotlib.use('agg')
console = Console()


class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, data_dict, transform=None, normaliser=2 ** 8 - 1, is_valid=False):
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
        self.opt = opt

        dir_this_GT = opt[DATA][GT_DIRPATH]
        dir_this_input = opt[DATA][INPUT_DIRPATH]

        console.log(f'GT Directory path: [red]{dir_this_GT}[/red]')
        console.log(f'Input Directory path: [red]{dir_this_input}[/red]')

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
        console.log('Dataset length: ', len(name_list_GT))
        for nn in range(len(name_list_GT)):
            self.train_ids.append([os.path.join(dir_this_GT, name_list_GT[nn]), os.path.join(
                dir_this_input, name_list_input[nn])])

    def __len__(self):
        """Returns the number of images in the dataset

        :returns: number of images in the dataset
        :rtype: Integer

        """
        return (len(self.train_ids))

    def applyTransformForOneImg(self, img, seed):
        height = img.shape[0]
        width = img.shape[1]

        # ─── APPLY CUSTOM TRANSFORM ──────────────────────────────────────
        cropFactor = self.opt[AUGMENTATION][CROP]
        resizeFactor = self.opt[AUGMENTATION][RESIZE]

        # crop the image:
        if cropFactor:
            random.seed(seed)
            assert type(cropFactor) == int

            rnd_h = random.randint(0, max(0, height - cropFactor))
            rnd_w = random.randint(0, max(0, width - cropFactor))
            img = img[rnd_h:rnd_h + cropFactor,
                  rnd_w:rnd_w + cropFactor, :]

        # resize the image:
        elif resizeFactor:
            # accept int and float:
            assert type(resizeFactor + 0.1) == float

            img = cv2.resize(img, (int(width / resizeFactor),
                                   int(height / resizeFactor)))

        img = img.astype(np.uint8)

        # ─── APPLY PYTORCH TRANSFORM: ────────────────────────────────────
        random.seed(seed)
        img = self.transform(img)

        # console.log('', log_locals=True)

        return img

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

        seed = random.randint(0, 100000)
        input_img = self.applyTransformForOneImg(input_img, seed)
        output_img = self.applyTransformForOneImg(output_img, seed)

        return {INPUT_IMG: input_img, OUTPUT_IMG: output_img,
                NAME: self.train_ids[idx][1]}


class DataLoader():

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialization function for the data loader

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
        """Initialization function for the data loader

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
                        self.data_dict[idx][INPUT_IMG] = None
                        self.data_dict[idx][OUTPUT_IMG] = None
                        idx_tmp = idx
                        idx += 1
                    else:
                        idx_tmp = img_id_to_idx_dict[img_id]

                    if "input" in root:  # change this to the name of your
                        # input data folder

                        input_img_filepath = file

                        self.data_dict[idx_tmp][INPUT_IMG] = root + \
                            "/" + input_img_filepath

                    elif ("output" in root):  # change this to the name of your
                        # output data folder

                        output_img_filepath = file

                        self.data_dict[idx_tmp][OUTPUT_IMG] = root + \
                            "/" + output_img_filepath

                else:

                    logging.debug("Excluding file with id: " + str(img_id))

        for idx, imgs in self.data_dict.items():
            assert (INPUT_IMG in imgs)
            assert (OUTPUT_IMG in imgs)

        return self.data_dict
