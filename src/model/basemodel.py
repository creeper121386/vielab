import os
import os.path as osp
from collections.abc import Iterable

import pytorch_lightning as pl
import util
import wandb
from globalenv import *


class BaseModel(pl.core.LightningModule):
    def __init__(self, opt, logger_img_group_names):
        '''
        logger_img_group_names: images group names in wandb logger. recommand: ['train', 'valid']
        '''

        super().__init__()
        self.save_hyperparameters(opt)
        console.log('Running initialization for BaseModel')

        self.train_img_dirpath = osp.join(opt[IMG_DIRPATH], TRAIN)
        util.mkdir(self.train_img_dirpath)
        if opt[VALID_DATA][INPUT]:
            self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)
            util.mkdir(self.valid_img_dirpath)

        self.opt = opt
        self.MODEL_WATCHED = False  # for wandb watching model

        assert isinstance(logger_img_group_names, Iterable)
        self.logger_image_buffer = {k: [] for k in logger_img_group_names}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items.pop("loss", None)
        return items

    def save_one_img_of_batch(self, batch, dirpath, fname):
        if not osp.exists(dirpath):
            console.log(f'Image dir path "{dirpath}" not exists. Created.')
            os.makedirs(dirpath)

        imgpath = osp.join(dirpath, fname)
        assert len(batch.shape) == 4
        img = util.saveTensorAsImg(batch[0], imgpath)

    def log_IOG_images(self, mode, step, fname, input_batch, output_batch, gt_batch):
        '''
        log_IOG_images means: log input, output and gt images to local disk and remote wandb logger.
        mode: TRAIN or VALID
        '''
        if step % self.opt[LOG_EVERY] == 0:
            fname = osp.basename(fname) + f'_epoch{self.current_epoch}_iter{step}.png'
            if mode == VALID:
                self.save_one_img_of_batch(output_batch, self.valid_img_dirpath, fname)
            elif mode == TRAIN:
                self.save_one_img_of_batch(output_batch, self.train_img_dirpath, fname)

            self.logger_buffer_add_img(mode, input_batch, mode, INPUT, fname)
            self.logger_buffer_add_img(mode, output_batch, mode, OUTPUT, fname)
            self.logger_buffer_add_img(mode, gt_batch, mode, GT, fname)
            self.commit_logger_buffer(mode)

    def logger_buffer_add_img(self, group_name, batch, *caption):
        if len(batch.shape) == 3:
            # input is not a batch, but a single image.
            batch = batch.unsqueeze(0)

        self.logger_image_buffer[group_name].append(
            wandb.Image(batch[0], caption='-'.join(caption))
        )

    def commit_logger_buffer(self, groupname):
        assert self.logger
        self.logger.experiment.log({
            groupname: self.logger_image_buffer[groupname]
        })

        # clear buffer after each commit for the next commit
        self.logger_image_buffer[groupname].clear()
