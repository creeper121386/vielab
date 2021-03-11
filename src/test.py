# -*- coding: utf-8 -*-

import hydra
import pytorch_lightning as pl
import torch
from data import ImagesDataset
from globalenv import *
from util import checkConfig, configLogging


#
#
# def testWithGT(opt, log_dirpath, img_dirpath, net):
#     # testdata = ImagesDataset(opt, data_dict=None, transform=transforms.Compose(
#     #     [transforms.ToPILImage(), transforms.ToTensor()]),
#     #                          normaliser=2 ** 8 - 1, is_valid=False)
#     #
#     # dataloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False,
#     #                                          num_workers=opt[DATALOADER_NUM_WORKER])
#     psnr_all = psnr_count = ssim_all = ssim_count = 0
#     for batch_num, data in enumerate(dataloader, 0):
#         '''
#         psnr_count += 1
#         if psnr_count <= 201:
#             continue
#         '''
#         #
#         # x, y, category = Variable(data[INPUT_IMG], requires_grad=False), \
#         #                  Variable(data[OUTPUT_IMG], requires_grad=False), \
#         #                  data[NAME]
#         #
#         # if CUDA_AVAILABLE:
#         #     x, y = x.cuda(), y.cuda()
#
#         # path_split = category[0].split('/')
#         # path_id = path_split[-1].split('.jpg')[0]
#
#         # with torch.no_grad():
#         #     output_dict = net(x)
#         #     output = torch.clamp(output_dict[OUTPUT], 0.0, 1.0)
#
#         # saveTensorAsImg(output, os.path.join(img_dirpath, path_id + '.jpg'))
#
#         # if PREDICT_ILLUMINATION in output_dict:
#         #     # import ipdb; ipdb.set_trace()
#         #     illumination_path = os.path.join(log_dirpath, PREDICT_ILLUMINATION)
#         #     if not os.path.exists(illumination_path):
#         #         os.makedirs(illumination_path)
#         #     saveTensorAsImg(output_dict[PREDICT_ILLUMINATION], os.path.join(
#         #         illumination_path, path_id + '.jpg'))
#
#         # ─── CALCULATE METRICS ───────────────────────────────────────────
#     #     output_ = output.clone().detach().cpu().numpy()
#     #     y_ = y.clone().detach().cpu().numpy()
#     #
#     #     psnr_this = ImageProcessing.compute_psnr(output_, y_, 1.0)
#     #     ssim_this = ImageProcessing.compute_ssim(output_, y_)
#     #     psnr_all += psnr_this
#     #     psnr_count += 1
#     #     ssim_all += ssim_this
#     #     ssim_count += 1
#     #
#     #     console.log(
#     #         f'Test [[{psnr_count}]] SSIM: {ssim_this:.4f}, PSNR: {psnr_this:.4f}')
#     # console.log(f'Average PSNR: {psnr_all * 1.0 / psnr_count}')
#     # console.log(f'Average SSIM: {ssim_all * 1.0 / ssim_count}')
#
#
# # count = 0
# #
# #
# # def evalWithoutGT(opt, log_dirpath, img_dirpath, net, path):
#     global count
#     count += 1
#     console.log(f'[[Eval {count}]] Now processing {path}')
#     x = torch.Tensor(Image.open(path))
#     fname = osp.basename(path)
#
#     with torch.no_grad():
#         output_dict = net(x)
#         output = torch.clamp(output_dict[OUTPUT], 0.0, 1.0)
#
#     saveTensorAsImg(output, os.path.join(img_dirpath, fname))
#
#     if PREDICT_ILLUMINATION in output_dict:
#         # import ipdb; ipdb.set_trace()
#         illumination_path = os.path.join(log_dirpath, PREDICT_ILLUMINATION)
#         if not os.path.exists(illumination_path):
#             os.makedirs(illumination_path)
#         saveTensorAsImg(output_dict[PREDICT_ILLUMINATION], os.path.join(
#             illumination_path, fname))
#     return output.clone().detach().cpu().numpy()


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = checkConfig(opt, TEST)
    opt[LOG_DIRPATH], opt[IMG_DIRPATH] = configLogging(TEST, opt)

    model = DeepLpfLitModel.load_from_checkpoint(opt[CHECKPOINT_PATH], opt=opt)
    console.log(f'Loading model from: {opt[CHECKPOINT_PATH]}')

    ds = ImagesDataset(
        opt, data_dict=None,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=opt[DATALOADER_NUM_WORKER]
    )

    trainer = pl.Trainer(
        gpus=opt[GPU],
        distributed_backend='dp',
    )
    trainer.test(model, dataloader)

    # # load data:
    # if opt[DATA][GT_DIRPATH]:
    #     testWithGT(opt, log_dirpath, img_dirpath, net)
    # else:
    #     for x in glob.glob(opt[DATA][INPUT_DIRPATH]):
    #         evalWithoutGT(opt, log_dirpath, img_dirpath, net, x)


if __name__ == "__main__":
    main()
