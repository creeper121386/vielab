# -*- coding: utf-8 -*-
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import hydra
import pytorch_lightning as pl
import torch
from data import ImagesDataset
from globalenv import *
from model.deeplpf import DeepLpfLitModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch.utils.tensorboard import SummaryWriter
from util import checkConfig, configLogging, parseAugmentation

writer = SummaryWriter()


@hydra.main(config_path='config', config_name="config")
def main(config):
    opt = checkConfig(config, TRAIN)
    console.log('Parameters:', opt, log_locals=False)

    opt[LOG_DIRPATH], opt[IMG_DIRPATH] = configLogging(TRAIN, opt)
    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    ## Loading data:
    transform = parseAugmentation(opt)
    training_dataset = ImagesDataset(
        opt, data_dict=None,
        transform=transform,
        normaliser=2 ** 8 - 1, is_valid=False
    )
    trainloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=opt[DATALOADER_NUM_WORKER]
    )
    console.log('Finish loading data.')

    # callbacks:
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt[LOG_DIRPATH],
        save_last=True,
        save_weights_only=True,
        filename='{epoch:}-{step}-{loss:.3f}',
        save_top_k=10,  # save 10 model
        monitor='loss',
        # period=opt[SAVE_MODEL_EVERY],
    )

    # logger:
    comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
        # save_dir='../',  # Optional
        project_name='vielab',  # Optional
        experiment_name=opt[EXPNAME]  # Optional
    )
    # train:
    if opt[CHECKPOINT_PATH]:
        model = DeepLpfLitModel.load_from_checkpoint(opt[CHECKPOINT_PATH], opt=opt)
        console.log(f'Loading model from: {opt[CHECKPOINT_PATH]}')
    else:
        model = DeepLpfLitModel(opt)
    trainer = pl.Trainer(
        gpus=opt[GPU],
        # auto_select_gpus=True,
        distributed_backend='dp',
        max_epochs=opt[NUM_EPOCH],
        logger=comet_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, trainloader)


if __name__ == "__main__":
    main()
