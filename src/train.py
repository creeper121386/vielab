# -*- coding: utf-8 -*-
import logging
import os

# import comet_ml
import hydra

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytorch_lightning as pl
import torch
from dataset import ImagesDataset
from globalenv import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from util import checkConfig, configLogging, server_chan_send
from data_aug import parseAugmentation

from model.model_zoo import MODEL_ZOO


@hydra.main(config_path='config', config_name="config")
def main(config):
    # config and logging:
    # config = init_config(config)
    opt = checkConfig(config, TRAIN)
    # tab_complete(opt)

    if opt[DEBUG]:
        debug_config = {
            DATALOADER_NUM_WORKER: 0,
            EXPNAME: DEBUG,
            LOG_EVERY: 1,
            NUM_EPOCH: 2
        }
        opt.update(debug_config)
        console.log('[red]>>>> WARN: You are in debug mode, update configs. <<<<[/red]')
        console.log(debug_config)
        console.log('[red]>>>> WARN: You are in debug mode, update configs. <<<<[/red]')

    console.log('Running config:', opt, log_locals=False)
    opt[LOG_DIRPATH], opt[IMG_DIRPATH] = configLogging(TRAIN, opt)
    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    # init model:
    ModelClass = MODEL_ZOO[opt[RUNTIME][MODELNAME]]
    if opt[CHECKPOINT_PATH]:
        model = ModelClass.load_from_checkpoint(opt[CHECKPOINT_PATH], opt=opt)
        console.log(f'Loading model from: {opt[CHECKPOINT_PATH]}')
    else:
        model = ModelClass(opt)

    # Loading data:
    transform = parseAugmentation(opt)
    training_dataset = ImagesDataset(opt, data_dict=None, ds_type=DATA, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt[BATCHSIZE],
        shuffle=True,
        num_workers=opt[DATALOADER_NUM_WORKER]
    )

    valid_loader = None
    if opt[VALID_DATA] and opt[VALID_DATA][INPUT]:
        valid_dataset = ImagesDataset(opt, data_dict=None, ds_type=VALID_DATA, transform=transform)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=opt[VALID_BATCHSIZE],
            shuffle=False,
            num_workers=opt[DATALOADER_NUM_WORKER]
        )
    console.log('Finish loading data.')

    # callbacks:
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt[LOG_DIRPATH],
        save_last=True,
        save_weights_only=True,
        filename='{epoch:}-{step}',
        save_top_k=10,  # save 10 model
        monitor=opt[CHECKPOINT_MONITOR],
    )

    # trainer logger:
    mylogger = WandbLogger(
        name=opt[EXPNAME],
        project='vielab',
        notes=None if not opt[COMMENT] else opt[COMMENT]
    )

    # init trainer:
    trainer = pl.Trainer(
        gpus=opt[GPU],
        distributed_backend='dp',
        # auto_select_gpus=True,
        max_epochs=opt[NUM_EPOCH],
        logger=mylogger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=opt[VALID_EVERY]
    )

    # training loop
    trainer.fit(model, trainloader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.log(e)
        server_chan_send('[Exception] Training stop.', str(e))
