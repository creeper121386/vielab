import sys
from glob import glob

import numpy as np
from PIL import Image
from globalenv import PSNR, SSIM
from util import calculate_psnr, calculate_ssim

'''
Calculate metrics (psnr, ssim) for 2 image sets 
usage: python metrics.py '<glob1>' '<glob2>'
'''

assert len(sys.argv) == 3
assert '~' not in sys.argv[1] and '~' not in sys.argv[2]
folder1 = glob(sys.argv[1])
folder2 = glob(sys.argv[2])

assert len(folder1) == len(folder2)
i = 0
metrics = {
    PSNR: 0,
    SSIM: 0
}
# ipdb.set_trace()
for x, y in zip(folder1, folder2):
    i += 1
    print(f'[{i}] Calc: {x} and {y}')
    im1 = np.array(Image.open(x))
    im2 = np.array(Image.open(y))

    if im1.shape != im2.shape:
        print(f'WARN: image shape mismatch: {im1.shape} != {im2.shape}. Resized.')
        im2.resize(tuple(im1.shape))

    psnr = calculate_psnr(im1, im2)
    ssim = calculate_ssim(im1, im2)

    # Use torch to compute, removed.
    # im1 = torch.tensor(im1).unsqueeze(0)
    # im2 = torch.tensor(im2).unsqueeze(0)
    # psnr = ImageProcessing.compute_psnr(im1, im2, 255)
    # ssim = ImageProcessing.compute_ssim(im1, im2)

    metrics[PSNR] += psnr
    metrics[SSIM] += ssim

    print(f'[{i}] PSNR: {psnr}, SSIM: {ssim} - [AVG] PSNR: {metrics[PSNR] / i}, SSIM: {metrics[SSIM] / i}')

print(f'[AVG] PSNR: {metrics[PSNR] / i}, SSIM: {metrics[SSIM] / i}')
