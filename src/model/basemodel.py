import os
import os.path as osp

import pytorch_lightning as pl
import util
import wandb
from globalenv import *


class BaseModel(pl.core.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)
        console.log('Running initialization for BaseModel')

        self.train_img_dirpath = osp.join(opt[IMG_DIRPATH], TRAIN)
        if opt[VALID_DATA][INPUT]:
            self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)

        self.iternum = 0
        self.epoch = 0
        self.opt = opt

        self.logger_image_buffer = []

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items.pop("loss", None)
        return items

    def log_img(self, batch, dirpath, fname):
        if not osp.exists(dirpath):
            console.log(f'Image dir path "{dirpath}" not exists. Created.')
            os.makedirs(dirpath)

        imgpath = osp.join(dirpath, fname)
        img = util.saveTensorAsImg(batch, imgpath)

    def add_img_to_logger_buffer(self, batch, suffix, fname):
        if len(batch.shape) == 3:
            # input is not a batch, but a single image.
            batch = batch.unsqueeze(0)

        self.logger_image_buffer.append(
            wandb.Image(batch[0], caption=f'{suffix}_{fname}')
        )

    def commit_logger_buffer(self):
        assert self.logger
        self.logger.experiment.log({
            'Images': self.logger_image_buffer
        })

        # clear buffer after each commit for the next commit
        self.logger_image_buffer.clear()

    def training_epoch_end(self, outputs):
        self.epoch += 1
