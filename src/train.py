# -*- coding: utf-8 -*-
import logging
import os
import traceback

import hydra

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytorch_lightning as pl
import torch
from dataset import ImagesDataset
from globalenv import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from toolbox.util import checkConfig, configLogging
from data_aug import parseAugmentation

from model.model_zoo import MODEL_ZOO
from toolbox.util import send_mail


@hydra.main(config_path='config', config_name="config")
def main(config):
    opt = checkConfig(config, TRAIN)

    # update debug config (if in debug mode)
    if opt[DEBUG]:
        debug_config = {
            DATALOADER_NUM_WORKER: 0,
            NAME: DEBUG,
            LOG_EVERY: 1,
            VALID_EVERY: 1,
            NUM_EPOCH: 2
        }
        opt.update(debug_config)
        console.log('[red]>>>> [[ WARN ]] You are in debug mode, update configs. <<<<[/red]')
        console.log(debug_config)
        console.log('[red]>>>> [[ WARN ]] You are in debug mode, update configs. <<<<[/red]')

    # logging
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
    training_dataset = ImagesDataset(opt, ds_type=DATA, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt[BATCHSIZE],
        shuffle=True,
        num_workers=opt[DATALOADER_NUM_WORKER],
        drop_last = True
    )

    valid_loader = None
    if opt[VALID_DATA] and opt[VALID_DATA][INPUT]:
        valid_dataset = ImagesDataset(opt, ds_type=VALID_DATA, transform=transform)
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
        name=opt[NAME],
        project='vielab',
        notes=None if not opt[COMMENT] else opt[COMMENT],
        # tags = ['model:' + opt[RUNTIME][MODELNAME], 'ds:' + opt[DATA][NAME]]
        tags=[opt[RUNTIME][MODELNAME], opt[DATA][NAME]],
        save_dir='../'
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
    return opt


if __name__ == "__main__":
    try:
        opt = main()
        send_mail(
            f'[ DONE ] Training finished: {opt[NAME]}',
            f'Running arguments: {opt}'
        )

    except Exception as e:
        console.log(e)
        # server_chan_send('[Exception] Training stop.', str(e))
        send_mail('[ ERR ] Training stop by exception.', str(traceback.format_exc()))
