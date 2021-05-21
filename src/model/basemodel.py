import os
import os.path as osp
import pathlib
from collections.abc import Iterable

import pytorch_lightning as pl
import torchvision
import util
import wandb
from globalenv import *


class BaseModel(pl.core.LightningModule):
    def __init__(self, opt, running_modes):
        '''
        logger_img_group_names: images group names in wandb logger. recommand: ['train', 'valid']
        '''

        super().__init__()
        self.save_hyperparameters(opt)
        console.log('Running initialization for BaseModel')

        if IMG_DIRPATH in opt:
            # in training mode.
            # if in test mode, configLogging is not called.
            if TRAIN in running_modes:
                self.train_img_dirpath = osp.join(opt[IMG_DIRPATH], TRAIN)
                util.mkdir(self.train_img_dirpath)
            if VALID in running_modes and opt[VALID_DATA][INPUT]:
                self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)
                util.mkdir(self.valid_img_dirpath)

        self.opt = opt
        self.MODEL_WATCHED = False  # for wandb watching model
        self.global_valid_step = 0

        assert isinstance(running_modes, Iterable)
        self.logger_image_buffer = {k: [] for k in running_modes}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items.pop("loss", None)
        return items

    def build_test_res_dir(self):
        assert self.opt[CHECKPOINT_PATH]
        modelpath = pathlib.Path(self.opt[CHECKPOINT_PATH])

        # when passing valid_ds, ignore ds.name and use valid_ds.name instead.
        ds_type = VALID_DATA if self.opt[VALID_DATA][INPUT] else DATA
        fname = modelpath.name + '@' + self.opt[ds_type][NAME]
        dirpath = modelpath.parent / TEST_RESULT_DIRNAME

        if (dirpath / fname).exists():
            if len(os.listdir(dirpath / fname)) == 0:
                # an existing but empty dir
                pass
            else:
                input_str = input(
                    f'[ WARN ] Result directory "" exists. Press ENTER to overwrite or input suffix to create a new one:\n> New name: {fname}.')
                if input_str == '':
                    console.log(f"[ WARN ] Overwrite test result: {fname}")
                    pass
                else:
                    fname += '.' + input_str
            # fname += '.new'

        dirpath /= fname
        util.mkdir(dirpath)
        return str(dirpath)

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
            # log_step = step  # to avoid valid log step = train log step
        elif mode == TRAIN:
            local_dirpath = self.train_img_dirpath
            step = self.global_step
            # log_step = None

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

            self.commit_logger_buffer(mode, step=None)

    def add_img_to_buffer(self, group_name, batch, *caption):
        if len(batch.shape) == 3:
            # when input is not a batch:
            batch = batch.unsqueeze(0)

        self.logger_image_buffer[group_name].append(
            wandb.Image(batch[0], caption='-'.join(caption))
        )

    def commit_logger_buffer(self, groupname, **kwargs):
        assert self.logger
        self.logger.experiment.log({
            groupname: self.logger_image_buffer[groupname]
        }, **kwargs)

        # clear buffer after each commit for the next commit
        self.logger_image_buffer[groupname].clear()
