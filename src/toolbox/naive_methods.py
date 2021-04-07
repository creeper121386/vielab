import argparse
import os.path as osp

import cv2
import numpy as np
import util
from glob2 import glob
from rich.console import Console as C
from rich.progress import track

c = C()

parser = argparse.ArgumentParser(description='Naive image enhancement methods')
parser.add_argument('--input', '-i', help='Glob path containing input images.')
parser.add_argument('--output', '-o', help='Output directory.')
parser.add_argument('--methods', '-m', help='Method to use.', nargs='+')

args = parser.parse_args()


class HistEq:
    def __init__(self):
        pass

    def process_one(self, img):
        # import ipdb; ipdb.set_trace()
        for i in range(3):
            one_channel_img = img[:, :, i]
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
        # img = cv2.equalizeHist(img)
        return img


class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def process_one(self, img):
        # import ipdb; ipdb.set_trace()
        for i in range(3):
            one_channel_img = img[:, :, i]
            img[:, :, i] = self.clahe.apply(img[:, :, i])

        return img


method_map = {
    'clahe': CLAHE,
    'he': HistEq
}



def run(method_name):
    processed_fnames = []
    imgs = glob(args.input)
    c.log(f'Method: {method_name}, Input Images:')
    c.log(imgs)
    model = method_map[method_name]()
    for x in track(imgs):
        img = cv2.imread(x)
        res = model.process_one(np.array(img, dtype=np.uint8))

        # save result img:
        fname = osp.basename(x)
        if fname not in processed_fnames:
            processed_fnames.append(fname)
        else:
            new_fname = fname + '.dup.png'
            processed_fnames.append(new_fname)
            c.log(f'[*] {fname} already exists, rename to {new_fname}')
            fname = new_fname

        dstdir = osp.join(args.output, method_name)
        dst = osp.join(dstdir, fname)
        # if not osp.exists(dstdir):
        #     os.makedirs(dstdir)
        util.mkdir(dstdir)
        cv2.imwrite(dst, res)


if __name__ == '__main__':
    if len(args.methods) == 1:
        run(args.methods[0])
    else:
        for x in args.methods:
            run(x)
