# -*- coding: utf-8 -*-
import random
from glob import glob

import cv2
import numpy as np
import torch
from globalenv import *


def load_from_glob_list(globs):
    if type(globs) == str:
        return sorted(glob(globs))

    elif type(globs) == list:
        res = []
        for g in globs:
            res.extend(glob(g))
        return sorted(res)
    else:
        raise TypeError(
            f'ERR: Argument `ds.GT` or `ds.input` has wrong type: expect `str` or `list` but get {type(globs)}')


def augment_one_img(img, seed, transform=None):
    img = img.astype(np.uint8)
    random.seed(seed)
    torch.manual_seed(seed)
    if transform:
        img = transform(img)
    return img


class ImagesDataset(torch.utils.data.Dataset):

    def __init__(self, opt, ds_type=DATA, transform=None):
        """Initialisation for the Dataset object

        :param transform: PyTorch image transformations to apply to the images
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.opt = opt

        gt_globs = opt[ds_type][GT]
        input_globs = opt[ds_type][INPUT]
        self.have_gt = True if gt_globs else False

        console.log(f'[[{ds_type}]] GT Directory path: [yellow]{gt_globs}[/yellow]')
        console.log(f'[[{ds_type}]] Input Directory path: [yellow]{input_globs}[/yellow]')

        # load input images:
        self.input_list = load_from_glob_list(input_globs)

        # load GT images:
        if self.have_gt:
            self.gt_list = load_from_glob_list(gt_globs)
            assert len(self.input_list) == len(self.gt_list)

        console.log(f'[[{ds_type}]] Dataset length: {self.__len__()}, Batch num: {self.__len__() // opt[BATCHSIZE]}')

    def __len__(self):
        return (len(self.input_list))

    def debug_save_item(self, input, gt):
        import util
        # home = os.environ['HOME']
        util.saveTensorAsImg(input, 'i.png')
        util.saveTensorAsImg(gt, 'o.png')

    def __getitem__(self, idx):
        """Returns a pair of images with the given identifier. This is lazy loading
        of data into memory. Only those image pairs needed for the current batch
        are loaded.

        :param idx: image pair identifier
        :returns: dictionary containing input and output images and their identifier
        :rtype: dictionary

        """
        res_item = {FPATH: self.input_list[idx]}

        # different seed for each item, but same for GT and INPUT in one item:
        seed = random.randint(0, 100000)

        input_img = cv2.imread(self.input_list[idx])[:, :, [2, 1, 0]]
        input_img = augment_one_img(input_img, seed, transform=self.transform)
        res_item[INPUT] = input_img

        if self.have_gt:
            gt_img = cv2.imread(self.gt_list[idx])[:, :, [2, 1, 0]]
            gt_img = augment_one_img(gt_img, seed, transform=self.transform)
            res_item[GT] = gt_img


        return res_item
