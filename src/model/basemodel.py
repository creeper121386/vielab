import os
import os.path as osp

import pytorch_lightning as pl
import util
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

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items.pop("loss", None)
        return items

    def log_img(self, batch, img_dirpath, logname, fname):
        if not osp.exists(img_dirpath):
            console.log(f'Image dir path "{img_dirpath}" not exists. Created.')
            os.makedirs(img_dirpath)
        imgpath = osp.join(img_dirpath, fname)
        img = util.saveTensorAsImg(batch, imgpath)

        assert self.logger
        self.logger.experiment.log_image(img, overwrite=False, name=f'{logname}_{fname}')

    def training_epoch_end(self, outputs):
        self.epoch += 1
