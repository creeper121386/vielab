# -*- coding: utf-8 -*-
import random
from glob import glob

import cv2
import numpy as np
import torch
from globalenv import *


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, data_dict, transform=None):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param transform: PyTorch image transformations to apply to the images
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.data_dict = data_dict
        self.opt = opt

        gt_glob = opt[DATA][GT_DIRPATH]
        input_glob = opt[DATA][INPUT_DIRPATH]
        self.have_gt = True if gt_glob else False

        console.log(f'GT Directory path: [yellow]{gt_glob}[/yellow]')
        console.log(f'Input Directory path: [yellow]{input_glob}[/yellow]')

        # load input images:
        self.input_list = glob(input_glob)

        # load GT images:
        if self.have_gt:
            self.gt_list = glob(gt_glob)
            assert len(self.input_list) == len(self.gt_list)

        console.log('Dataset length: ', len(self.input_list))

    def __len__(self):
        return (len(self.input_list))

    def augment_one_img(self, img, seed):
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
        res_item = {NAME: self.input_list[idx]}

        seed = random.randint(0, 100000)
        input_img = cv2.imread(self.input_list[idx])[:, :, [2, 1, 0]]
        input_img = self.augment_one_img(input_img, seed)
        res_item[INPUT_IMG] = input_img

        if self.have_gt:
            output_img = cv2.imread(self.gt_list[idx])[:, :, [2, 1, 0]]
            output_img = self.augment_one_img(output_img, seed)
            res_item[OUTPUT_IMG] = output_img

        return res_item
