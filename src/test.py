# -*- coding: utf-8 -*-
import logging

import hydra
import torch
import torchvision
from data_aug import parseAugmentation
from dataset import ImagesDataset
from globalenv import *
from model.model_zoo import MODEL_ZOO
from pytorch_lightning import Trainer
from toolbox.util import checkConfig, configLogging


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = checkConfig(opt, TEST)
    opt[LOG_DIRPATH], opt[IMG_DIRPATH] = configLogging(TEST, opt)

    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    ModelClass = MODEL_ZOO[opt[RUNTIME][MODELNAME]]

    assert opt[CHECKPOINT_PATH]
    model = ModelClass.load_from_checkpoint(opt[CHECKPOINT_PATH], opt=opt)
    console.log(f'Loading model from: {opt[CHECKPOINT_PATH]}')

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

    ds = ImagesDataset(opt, data_dict=None, ds_type=DATA, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=opt[DATALOADER_NUM_WORKER]
    )
    trainer = Trainer(gpus=opt[GPU], distributed_backend='dp')
    trainer.test(model, dataloader)


if __name__ == "__main__":
    main()
