# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import os
import os.path
import time
from rich.console import Console

import cv2
import matplotlib
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

import model
from data_rx import Dataset

matplotlib.use('agg')
console = Console()
# np.set_printoptions(threshold=np.nan)


def main():
    # ─── CONFIG LOGGING ─────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "./log_" + timestamp
    os.mkdir(log_dirpath)
    handlers = [logging.FileHandler(
        log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)
    del timestamp, handlers
    # ───  ───────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")
    parser.add_argument(
        "--num_epoch", type=int, required=False, help="Number of epoches (default 5000)", default=100000)
    parser.add_argument(
        "--gpu", type=int, required=False, help="which gpu to use.", default=0)
    parser.add_argument(
        "--valid_every", type=int, required=False, help="Number of epoches after which to compute validation accuracy",
        default=500)
    parser.add_argument(
        "--save_every", type=int, required=False, help="Number of epoches after which to save the model.",
        default=1)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file to continue train", default=None)
    # parser.add_argument(
    #     "--GT_dirpath", required=True,
    #     help="Directory containing ground truth images for training.")
    # parser.add_argument(
    #     "--input_dirpath", required=True,
    #     help="Directory containing input images for training")
    args = parser.parse_args()
    num_epoch = args.num_epoch
    valid_every = args.valid_every
    checkpoint_filepath = args.checkpoint_filepath
    save_every = args.save_every

    torch.cuda.set_device(args.gpu)
    console.log('Current cuda device:', torch.cuda.current_device())
    del parser, args
    console.log('Paramters:', log_locals=True)

    training_dataset = Dataset(data_dict=None, transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor()]),
        normaliser=2 ** 8 - 1, is_valid=False)

    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True,
                                                       num_workers=4)

    net = model.DeepLPFNet()
    start_epoch = 1

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

    if (checkpoint_filepath is not None):
        para = torch.load(checkpoint_filepath,
                          map_location=lambda storage, location: storage)
        net.load_state_dict(para)

        base_name = os.path.basename(checkpoint_filepath)
        start_epoch = base_name.split('_')[-1]
        start_epoch = start_epoch.split('.pth')[0]
        start_epoch = int(start_epoch)

    console.log('Epoch start from:', start_epoch)
    count = 0
    for epoch in range(start_epoch, num_epoch, 1):

        # Train loss
        examples = 0.0
        running_loss = 0.0

        for batch_num, data in enumerate(training_data_loader, 0):

            input_batch, gt_batch, category = Variable(data['input_img'], requires_grad=False).cuda(), \
                Variable(data['output_img'],
                         requires_grad=False).cuda(), data['name']

            output = net(input_batch)
            # import ipdb; ipdb.set_trace()
            output = torch.clamp(
                output, 0.0, 1.0)

            if count % 500 == 0:
                output_image11 = output[0, :, :, :]
                output_image11 = output_image11.permute(1, 2, 0)
                output_image11 = output_image11.clone().detach().cpu().numpy()
                output_image11 = output_image11 * 255.0
                output_image11 = output_image11[:, :, [2, 1, 0]]
                cv2.imwrite('test2.jpg', output_image11.astype(np.uint8))

            loss = criterion(output, gt_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            examples += batch_size

            count += 1
            logging.info('[%d] train loss: %.15f' %
                         (epoch + 1, running_loss / examples))

        if count % save_every == 0:
            snapshot_prefix = os.path.join(log_dirpath, 'deep_lpf')
            snapshot_path = snapshot_prefix + "_" + str(epoch) + ".pth"
            torch.save(net.state_dict(), snapshot_path)

    snapshot_prefix = os.path.join(log_dirpath, 'deep_lpf')
    snapshot_path = snapshot_prefix + "_" + str(num_epoch)
    torch.save(net.state_dict(), snapshot_path)


if __name__ == "__main__":
    main()
