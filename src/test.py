# -*- coding: utf-8 -*-
import logging
import os
import time

import hydra
import torch
import torchvision
from data_aug import parseAugmentation
from dataset import ImagesDataset
from globalenv import *
from model.model_zoo import parse_model_class
from pytorch_lightning import Trainer
from util import parse_config


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = parse_config(opt, TEST)
    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    # passing `valid_ds` will overwrite `ds`
    if opt[VALID_DATA][INPUT]:
        console.log('[[ WARN ]] Found `valid_ds` in arguments. The value of `ds` will be overwrited by `valid_ds`.')
        opt[DATA] = opt[VALID_DATA]

    ModelClass = parse_model_class(opt[RUNTIME][MODELNAME])

    ckpt = opt[CHECKPOINT_PATH]
    assert ckpt
    model = ModelClass.load_from_checkpoint(ckpt, opt=opt)
    model.opt[IMG_DIRPATH] = model.build_test_res_dir()
    console.log(f'Loading model from: {ckpt}')

    transform = parseAugmentation(opt)
    if opt[AUGMENTATION][CROP]:
        console.log(
            f'[yellow]WRAN: You are testing the model but aug.crop is {opt[AUGMENTATION][CROP]}. Ignore the crop? (Y/n) [/yellow]')
        res = input()
        if res == 'n':
            console.log('Testing with cropped data...')
        else:
            console.log('Ignore and set transform=None...')
            transform = torchvision.transforms.ToTensor()

    ds = ImagesDataset(opt, ds_type=DATA, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=opt[DATALOADER_NUM_WORKER]
    )
    trainer = Trainer(gpus=opt[GPU], distributed_backend='dp')

    # test.
    beg = time.time()
    trainer.test(model, dataloader)
    console.log(f'[ TIMER ] Total time usage: {time.time() - beg}, #Dataset sample num: {len(ds)}')

    console.log('[ PATH ] The results are in :')
    console.log(model.opt[IMG_DIRPATH])

if __name__ == "__main__":
    main()
