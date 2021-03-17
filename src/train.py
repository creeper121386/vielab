# -*- coding: utf-8 -*-
import logging
import os
import os.path as osp

import comet_ml
import hydra
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytorch_lightning as pl
import torch
from data import ImagesDataset
from globalenv import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from util import checkConfig, configLogging, parseAugmentation

from model.model_zoo import MODEL_ZOO


@hydra.main(config_path='config', config_name="config")
def main(config):
    # config and logging:
    # config = init_config(config)
    opt = checkConfig(config, TRAIN)
    # tab_complete(opt)
    console.log('Running config:', opt, log_locals=False)
    opt[LOG_DIRPATH], opt[IMG_DIRPATH] = configLogging(TRAIN, opt)
    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    # init model:
    modelname = opt[RUNTIME][MODELNAME]
    if modelname not in MODEL_ZOO:
        raise RuntimeError(f'ERR: Model {modelname} not found. Please change the argument `runtime.modelname`')
    ModelClass = MODEL_ZOO[modelname]
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
    if opt[VALID_DATA]:
        # TODO: 这里valid_ds是手动用yaml解析的。有无办法让hydra自动解析啊
        opt[VALID_DATA] = yaml.load(open(osp.join(CONFIG_DIRPATH, DATA, f'{opt[VALID_DATA]}.yaml'), 'r').read(),
                                    Loader=yaml.FullLoader)

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
    comet_ml.config.DEBUG = False
    comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
        # save_dir='../',  # used in local mode
        project_name='vielab',  # Optional
        experiment_name=opt[EXPNAME]  # Optional
    )

    # init trainer:
    trainer = pl.Trainer(
        gpus=opt[GPU],
        distributed_backend='dp',
        # auto_select_gpus=True,
        max_epochs=opt[NUM_EPOCH],
        logger=comet_logger,
        callbacks=[checkpoint_callback],
    )

    # training loop
    trainer.fit(model, trainloader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
