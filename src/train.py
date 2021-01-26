# -*- coding: utf-8 -*-
import argparse
import logging
import os
import os.path as osp
from globalenv import *

import cv2
# import matplotlib
import numpy as np
import torch
from util import parseConfig, saveTensorAsImg, configLogging
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

import model
from data import Dataset

# matplotlib.use('agg')
# np.set_printoptions(threshold=np.nan)


def get_transform(opt):
    transformConfig = opt[TRANSFORMS]

    transformList = [transforms.ToPILImage(), ]
    if transformConfig[HORIZON_FLIP]:
        transformList.append(transforms.RandomHorizontalFlip())

    elif transformConfig[VERTICAL_FLIP]:
        transformList.append(transforms.RandomVerticalFlip())

    transformList.append(transforms.ToTensor())
    return transforms.Compose(transformList)


def main():
    # ─── PARSE CONFIG ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")
    parser.add_argument(
        "--configpath", '-c', required=True, help="yml config file path")
    args = parser.parse_args()
    opt = parseConfig(args.configpath, 'train')
    # opt[EXPNAME] = osp.basename(osp.splitext(args.configpath)[0])

    num_epoch = opt[NUM_EPOCH]
    valid_every = opt[VALID_EVERY]
    checkpoint_filepath = opt[CHECKPOINT_FILEPATH]
    save_every = opt[SAVE_MODEL_EVERY]
    log_every = opt[LOG_EVERY]

    torch.cuda.set_device(opt[GPU])
    console.log('Current cuda device:', torch.cuda.current_device())

    del parser, args
    console.log('Paramters:', opt, log_locals=False)

    # ─── CONFIG LOGGING ─────────────────────────────────────────────────────────────
    log_dirpath, img_dirpath = configLogging('train', opt)

    # ─── LOAD DATA ──────────────────────────────────────────────────────────────────
    transform = get_transform(opt)

    training_dataset = Dataset(opt, data_dict=None, transform=transform,
                               normaliser=2 ** 8 - 1, is_valid=False)

    trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True,
                                              num_workers=4)
    net = model.DeepLPFNet()
    start_epoch = 0

    # ─── PRINT NET LAYERS ───────────────────────────────────────────────────────────
    # logging.info('######### Network created #########')
    # logging.info('Architecture:\n' + str(net))
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # ───  ───────────────────────────────────────────────────────────────────────────

    criterion = model.DeepLPFLoss(ssim_window_size=5)

    console.log('Model initialization finished.')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

    best_valid_psnr = 0.0

    alpha = 0.0
    optimizer.zero_grad()
    net.train()

    running_loss = 0.0
    examples = 0
    psnr_avg = 0.0
    ssim_avg = 0.0
    batch_size = 1
    net.cuda()

    if checkpoint_filepath:
        para = torch.load(checkpoint_filepath,
                          map_location=lambda storage, location: storage)
        net.load_state_dict(para)

        base_name = osp.basename(checkpoint_filepath)
        start_epoch = base_name.split('_')[-1]
        start_epoch = start_epoch.split('.pth')[0]
        start_epoch = int(start_epoch)

    console.log('Epoch start from:', start_epoch)
    iternum = 0
    for epoch in range(start_epoch, num_epoch, 1):

        # Train loss
        examples = 0.0
        running_loss = 0.0

        for batch_num, data in enumerate(trainloader, 0):
            input_batch, gt_batch, category = Variable(data['input_img'], requires_grad=False).cuda(), \
                Variable(data['output_img'],
                         requires_grad=False).cuda(), data['name']

            saveTensorAsImg(input_batch, 'debug/i.png')
            saveTensorAsImg(gt_batch, 'debug/o.png')

            output = net(input_batch)
            output = torch.clamp(
                output, 0.0, 1.0)

            if iternum % 500 == 0:
                saveTensorAsImg(output, osp.join(
                    img_dirpath, f'epoch{epoch}_iter{iternum}.png'))

            loss = criterion(output, gt_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            examples += batch_size

            iternum += 1

            if iternum % log_every == 0:
                logging.info('[%d] iter: %d, train loss: %.10f' %
                             (epoch + 1, iternum, running_loss / examples))

        if epoch % save_every == 0:
            snapshot_prefix = osp.join(log_dirpath, 'deep_lpf')
            snapshot_path = snapshot_prefix + "_" + str(epoch) + ".pth"
            torch.save(net.state_dict(), snapshot_path)

    snapshot_prefix = osp.join(log_dirpath, 'deep_lpf')
    snapshot_path = snapshot_prefix + "_" + str(num_epoch)
    torch.save(net.state_dict(), snapshot_path)


if __name__ == "__main__":
    main()
