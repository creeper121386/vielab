# -*- coding: utf-8 -*-
import hydra
import torch
from data import ImagesDataset
from globalenv import *
from model.deeplpf import DeepLpfLitModel
from pytorch_lightning import Trainer
from util import checkConfig, configLogging, parseAugmentation


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = checkConfig(opt, TEST)
    opt[LOG_DIRPATH], opt[IMG_DIRPATH] = configLogging(TEST, opt)

    model = DeepLpfLitModel.load_from_checkpoint(opt[CHECKPOINT_PATH], opt=opt)
    console.log(f'Loading model from: {opt[CHECKPOINT_PATH]}')

    transform = parseAugmentation(opt)
    ds = ImagesDataset(opt, data_dict=None, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=opt[DATALOADER_NUM_WORKER]
    )
    trainer = Trainer(gpus=opt[GPU], distributed_backend='dp')

    trainer.test(model, dataloader)


if __name__ == "__main__":
    main()
