# -*- coding: utf-8 -*-
import logging
import os

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
def main(opt):
    opt = checkConfig(opt, TRAIN)

    # # valid_every = opt[VALID_EVERY]
    # save_every = opt[SAVE_MODEL_EVERY]
    console.log('Parameters:', opt, log_locals=False)

    log_dirpath, img_dirpath = configLogging(TRAIN, opt)
    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    ## Loading data:
    transform = parseAugmentation(opt)
    training_dataset = ImagesDataset(opt, data_dict=None, transform=transform,
                                     normaliser=2 ** 8 - 1, is_valid=False)
    trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True,
                                              num_workers=4)
    console.log('Finish loading data.')

    # callbacks:
    # TODO: 为啥存不上checkpoint啊啊啊啊啊啊啊啊啊啊啊啊
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dirpath,
        save_last=True,
        save_weights_only=True,
        period=opt[SAVE_MODEL_EVERY]
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
    model = DeepLpfLitModel(opt)
    trainer = pl.Trainer(
        gpus=opt[GPU],
        distributed_backend='dp',
        max_epochs=opt[NUM_EPOCH],
        logger=comet_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, trainloader)

    # best_valid_psnr = 0.0

    # alpha = 0.0
    # optimizer.zero_grad()
    # net.train()

    # running_loss = 0.0
    # examples = 0
    # psnr_avg = 0.0
    # ssim_avg = 0.0
    # batch_size = 1
    # if model_path:
    #     para = torch.load(model_path,
    #                       map_location=lambda storage, location: storage)
    #     net.load_state_dict(para)
    #
    #     base_name = osp.basename(model_path)
    #     start_epoch = base_name.split('_')[-1]
    #     start_epoch = start_epoch.split('.pth')[0]
    #     start_epoch = int(start_epoch)
    #
    # console.log('Epoch start from:', start_epoch)
    # iter_num = 0
    # for epoch in range(start_epoch, num_epoch, 1):
    #
    #     # Train loss
    #     examples = 0.0
    #     running_loss = 0.0
    #
    #     losses = {
    #         SSIM_LOSS: 0,
    #         L1_LOSS: 0,
    #         LTV_LOSS: 0,
    #         COS_LOSS: 0
    #     }
    #     for batch_num, data in enumerate(trainloader, 0):
    #         input_batch, gt_batch, category = Variable(data[INPUT_IMG], requires_grad=False), \
    #                                           Variable(data[OUTPUT_IMG],
    #                                                    requires_grad=False), data[NAME]
    #         if CUDA_AVAILABLE:
    #             input_batch, gt_batch = input_batch.cuda(), gt_batch.cuda()
    #
    #         # saveTensorAsImg(input_batch, 'debug/i.png')
    #         # saveTensorAsImg(gt_batch, 'debug/o.png')
    #
    #         output_dict = net(input_batch)
    #
    #         output = torch.clamp(
    #             output_dict[OUTPUT], 0.0, 1.0)
    #
    #         # ─── SAVE LOG ────────────────────────────────────────────────────
    #         if iter_num % 500 == 0:
    #             saveTensorAsImg(output, osp.join(
    #                 img_dirpath, f'epoch{epoch}_iter{iter_num}.png'))
    #
    #             if PREDICT_ILLUMINATION in output_dict:
    #                 illumination_path = os.path.join(
    #                     log_dirpath, PREDICT_ILLUMINATION)
    #                 if not os.path.exists(illumination_path):
    #                     os.makedirs(illumination_path)
    #                 saveTensorAsImg(output_dict[PREDICT_ILLUMINATION], os.path.join(
    #                     illumination_path, f'epoch{epoch}_iter{iter_num}.png'))
    #
    #         loss = criterion(output_dict, gt_batch)
    #
    #         this_losses = criterion.get_current_loss()
    #         for k in losses:
    #             if this_losses[k] is not None:
    #                 losses[k] += this_losses[k]
    #             else:
    #                 losses[k] = STRING_FALSE
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.data[0]
    #         examples += batch_size
    #
    #         iter_num += 1
    #
    #         # ─── LOGGING LOSSES ──────────────────────────────────────────────
    #         info = '[%d] iter: %d, ' % (epoch + 1, iter_num)
    #         for k in losses:
    #             if type(losses[k]) != torch.Tensor:
    #                 loss_value = losses[k]
    #                 info += f'{k}: {loss_value}, '
    #             else:
    #                 loss_value = float(losses[k] / examples)
    #                 info += f'{k}: {loss_value:.5f}, '
    #
    #                 writer.add_scalar('Train-Loss/' + k, loss_value, iter_num)
    #
    #         writer.add_scalar('Train-Loss/TotalLoss',
    #                           running_loss / examples, iter_num)
    #         info += 'Total loss: %.10f' % (running_loss / examples)
    #
    #         if iter_num % log_every == 0:
    #             logging.info(info)
    #
    #     if epoch % save_every == 0:
    #         snapshot_prefix = osp.join(log_dirpath, opt[RUNTIME][MODELNAME])
    #         snapshot_path = snapshot_prefix + "_" + str(epoch) + ".pth"
    #         torch.save(net.state_dict(), snapshot_path)
    #
    # snapshot_prefix = osp.join(log_dirpath, opt[RUNTIME][MODELNAME])
    # snapshot_path = snapshot_prefix + "_" + str(num_epoch)
    # torch.save(net.state_dict(), snapshot_path)
    #
    # writer.close()


if __name__ == "__main__":
    main()
