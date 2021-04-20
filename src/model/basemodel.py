import os.path as osp
from collections.abc import Iterable

import pytorch_lightning as pl
import torchvision
import wandb
from globalenv import *
from toolbox import util


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
        self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)
        util.mkdir(self.valid_img_dirpath)

        if opt[VALID_DATA][INPUT]:
            self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)
            util.mkdir(self.valid_img_dirpath)

        self.opt = opt
        self.MODEL_WATCHED = False  # for wandb watching model
        self.global_valid_step = 0

        assert isinstance(logger_img_group_names, Iterable)
        self.logger_image_buffer = {k: [] for k in logger_img_group_names}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items.pop("loss", None)
        return items

    def save_one_img_of_batch(self, batch, dirpath, fname):
        util.mkdir(dirpath)

        imgpath = osp.join(dirpath, fname)
        assert len(batch.shape) == 4
        # img = util.saveTensorAsImg(batch[0], imgpath)
        torchvision.utils.save_image(batch[0], imgpath)

    def log_images_dict(self, mode, input_fname, img_batch_dict):
        '''
        log input, output and gt images to local disk and remote wandb logger.
        mode: TRAIN or VALID
        '''
        if mode == VALID:
            local_dirpath = self.valid_img_dirpath
            step = self.global_valid_step
            if self.global_valid_step == 0:
                console.log(
                    'WARN: Found global_valid_step=0. Maybe you foget to increase `self.global_valid_step` in `self.validation_step`?')
        elif mode == TRAIN:
            local_dirpath = self.train_img_dirpath
            step = self.global_step

        if step % self.opt[LOG_EVERY] == 0:
            input_fname = osp.basename(input_fname) + f'_epoch{self.current_epoch}_step{step}.png'

            for name, batch in img_batch_dict.items():
                if batch is None or batch is False:
                    # image is None or False, skip.
                    continue

                # save local image:
                self.save_one_img_of_batch(
                    batch,
                    osp.join(local_dirpath, name),  # e.g. ../train_log/train/output
                    input_fname)

                # save remote image:
                self.add_img_to_buffer(mode, batch, mode, name, input_fname)

            self.commit_logger_buffer(mode)

    def add_img_to_buffer(self, group_name, batch, *caption):
        if len(batch.shape) == 3:
            # when input is not a batch:
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
