# -*- coding: utf-8 -*-
import logging
import os
import os.path as osp

import hydra
import torch
import torch.optim as optim
from data import Dataset
from globalenv import *
from model.DeepLPF import DeepLPFNet, DeepLPFLoss
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from util import checkConfig, saveTensorAsImg, configLogging, parseAugmentation

writer = SummaryWriter()


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = checkConfig(opt, TRAIN)

    num_epoch = opt[NUM_EPOCH]
    # valid_every = opt[VALID_EVERY]
    model_path = opt[MODEL_PATH]
    save_every = opt[SAVE_MODEL_EVERY]
    log_every = opt[LOG_EVERY]

    if CUDA_AVAILABLE:
        torch.cuda.set_device(opt[GPU])
        console.log('Current cuda device:', torch.cuda.current_device())

    console.log('Parameters:', opt, log_locals=False)

    log_dirpath, img_dirpath = configLogging(TRAIN, opt)

    ## Loading data:
    transform = parseAugmentation(opt)
    training_dataset = Dataset(opt, data_dict=None, transform=transform,
                               normaliser=2 ** 8 - 1, is_valid=False)

    trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True,
                                              num_workers=4)
    net = DeepLPFNet(opt)
    start_epoch = 0

    # ─── PRINT NET LAYERS ───────────────────────────────────────────────────────────
    # logging.info('######### Network created #########')
    # logging.info('Architecture:\n' + str(net))
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # ───  ───────────────────────────────────────────────────────────────────────────

    criterion = DeepLPFLoss(opt, ssim_window_size=5)

    console.log('Model initialization finished.')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

    # best_valid_psnr = 0.0

    # alpha = 0.0
    optimizer.zero_grad()
    net.train()

    # running_loss = 0.0
    # examples = 0
    # psnr_avg = 0.0
    # ssim_avg = 0.0
    batch_size = 1
    if CUDA_AVAILABLE:
        net.cuda()

    if model_path:
        para = torch.load(model_path,
                          map_location=lambda storage, location: storage)
        net.load_state_dict(para)

        base_name = osp.basename(model_path)
        start_epoch = base_name.split('_')[-1]
        start_epoch = start_epoch.split('.pth')[0]
        start_epoch = int(start_epoch)

    console.log('Epoch start from:', start_epoch)
    iter_num = 0
    for epoch in range(start_epoch, num_epoch, 1):

        # Train loss
        examples = 0.0
        running_loss = 0.0

        losses = {
            SSIM_LOSS: 0,
            L1_LOSS: 0,
            LTV_LOSS: 0,
            COS_LOSS: 0
        }
        for batch_num, data in enumerate(trainloader, 0):
            input_batch, gt_batch, category = Variable(data[INPUT_IMG], requires_grad=False), \
                                              Variable(data[OUTPUT_IMG],
                                                       requires_grad=False), data[NAME]
            if CUDA_AVAILABLE:
                input_batch, gt_batch = input_batch.cuda(), gt_batch.cuda()

            # saveTensorAsImg(input_batch, 'debug/i.png')
            # saveTensorAsImg(gt_batch, 'debug/o.png')

            output_dict = net(input_batch)

            output = torch.clamp(
                output_dict[OUTPUT], 0.0, 1.0)

            # ─── SAVE LOG ────────────────────────────────────────────────────
            if iter_num % 500 == 0:
                saveTensorAsImg(output, osp.join(
                    img_dirpath, f'epoch{epoch}_iter{iter_num}.png'))

                if PREDICT_ILLUMINATION in output_dict:
                    illumination_path = os.path.join(
                        log_dirpath, PREDICT_ILLUMINATION)
                    if not os.path.exists(illumination_path):
                        os.makedirs(illumination_path)
                    saveTensorAsImg(output_dict[PREDICT_ILLUMINATION], os.path.join(
                        illumination_path, f'epoch{epoch}_iter{iter_num}.png'))

            loss = criterion(output_dict, gt_batch)

            this_losses = criterion.get_currnet_loss()
            for k in losses:
                if this_losses[k] is not None:
                    losses[k] += this_losses[k]
                else:
                    losses[k] = STRING_FALSE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            examples += batch_size

            iter_num += 1

            # ─── LOGGING LOSSES ──────────────────────────────────────────────
            info = '[%d] iter: %d, ' % (epoch + 1, iter_num)
            for k in losses:
                if type(losses[k]) != torch.Tensor:
                    loss_value = losses[k]
                    info += f'{k}: {loss_value}, '
                else:
                    loss_value = float(losses[k] / examples)
                    info += f'{k}: {loss_value:.5f}, '

                    writer.add_scalar('Train-Loss/' + k, loss_value, iter_num)

            writer.add_scalar('Train-Loss/TotalLoss',
                              running_loss / examples, iter_num)
            info += 'Total loss: %.10f' % (running_loss / examples)

            if iter_num % log_every == 0:
                logging.info(info)

        if epoch % save_every == 0:
            snapshot_prefix = osp.join(log_dirpath, opt[RUNTIME][MODELNAME])
            snapshot_path = snapshot_prefix + "_" + str(epoch) + ".pth"
            torch.save(net.state_dict(), snapshot_path)

    snapshot_prefix = osp.join(log_dirpath, opt[RUNTIME][MODELNAME])
    snapshot_path = snapshot_prefix + "_" + str(num_epoch)
    torch.save(net.state_dict(), snapshot_path)

    writer.close()


if __name__ == "__main__":
    main()
