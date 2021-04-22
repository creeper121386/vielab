import os.path as osp
import sys
from glob import glob

import numpy as np
from PIL import Image
from globalenv import PSNR, SSIM, console, METRICS_LOG_DIRPATH
from util import calculate_psnr, calculate_ssim

'''
Calculate metrics (psnr, ssim) for 2 image sets 
usage: python metrics.py '<glob1>' '<glob2>'
'''

assert len(sys.argv) == 3
assert '~' not in sys.argv[1] and '~' not in sys.argv[2]
file = osp.join(METRICS_LOG_DIRPATH, f'metrics-{sys.argv[1].replace("/", ".")}-and-{sys.argv[2].replace("/", ".")}.csv')
folder1 = glob(sys.argv[1])
folder2 = glob(sys.argv[2])

assert len(folder1) == len(folder2)
i = 0
metrics = {
    PSNR: 0,
    # SSIM: 0
}
console.log(f'[*] Metrics: {list(metrics.keys())}')

f = open(file, 'w')
f.write('fname1,fname2,' + ','.join(list(metrics.keys())) + '\n')
for x, y in zip(folder1, folder2):
    i += 1
    console.log(f'[[{i}]] Now running: {x} and {y}')
    im1 = np.array(Image.open(x))
    im2 = np.array(Image.open(y))
    f.write(f'{x},{y}')

    if im1.shape != im2.shape:
        console.log(f'WARN: image shape mismatch: {im1.shape} != {im2.shape}. Resized.')
        im2.resize(tuple(im1.shape))

    if PSNR in metrics:
        psnr = calculate_psnr(im1, im2)
        metrics[PSNR] += psnr
        console.log(f'[[{i}]] PSNR: {psnr}, [[AVG]] PSNR: {metrics[PSNR] / i}')
        f.write(',' + str(psnr))

    if SSIM in metrics:
        ssim = calculate_ssim(im1, im2)
        metrics[SSIM] += ssim
        console.log(f'[[{i}]] SSIM: {ssim} - [[AVG]] SSIM: {metrics[SSIM] / i}')
        f.write(',' + str(ssim))

    f.write('\n')

    # Use torch (GPU) to compute, removed.
    # im1 = torch.tensor(im1).unsqueeze(0)
    # im2 = torch.tensor(im2).unsqueeze(0)
    # psnr = ImageProcessing.compute_psnr(im1, im2, 255)
    # ssim = ImageProcessing.compute_ssim(im1, im2)

# print(f'[AVG] PSNR: {metrics[PSNR] / i}, SSIM: {metrics[SSIM] / i}')
