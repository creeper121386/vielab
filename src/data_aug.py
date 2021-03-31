# import cv2
from collections.abc import Iterable

import numpy as np
import torchvision.transforms.functional as F
from globalenv import *
from torchvision import transforms


class RandomLightnessAdjustment:
    def __call__(self, img):
        '''
        img: PIL image.
        '''
        factor = np.random.uniform(0.6, 1.4)
        # img = np.array(img)
        # type_info = np.iinfo(img.dtype)
        # res = np.clip(img.astype(np.float) + bias, type_info.min, type_info.max).astype(img.dtype)
        # return Image.fromarray(res)
        return F.adjust_brightness(img, factor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomContrastAdjustment:
    def __call__(self, img):
        '''
        img: PIL image.
        '''
        factor = np.random.uniform(0.8, 1.2)
        return F.adjust_contrast(img, factor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Downsample:
    def __init__(self, downsample_factor=None):
        self.downsample_factor = downsample_factor

        if isinstance(self.downsample_factor, Iterable):
            assert len(downsample_factor) == 2

    def __call__(self, img):
        '''
        img: passed by the previous transforms. PIL iamge or np.ndarray
        '''
        if isinstance(self.downsample_factor, Iterable):
            if -1 in self.downsample_factor:
                w_scale = img.size[0] / self.downsample_factor[0]
                h_scale = img.size[1] / self.downsample_factor[1]

                # choose the correct one
                scale = max(w_scale, h_scale)
                new_size = [
                    int(img.size[0] / scale),
                    int(img.size[1] / scale)
                ]
            else:
                new_size = self.downsample_factor

        elif type(self.downsample_factor + 0.1) == float:
            # PIL.Image.resize accepts [H, W];
            # while cv2.resize and torchvision.transforms.Resize accepts [W, H]
            new_size = [
                int(img.size[0] / self.downsample_factor),
                int(img.size[1] / self.downsample_factor)
            ]
        else:
            raise RuntimeError(f'ERR: Wrong config aug.downsample: {self.downsample_factor}')

        img = img.resize(new_size)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'({self.downsample_factor})'


def parseAugmentation(opt):
    '''
    return: pytorch composed transform
    '''
    aug_config = opt[AUGMENTATION]
    aug_list = [transforms.ToPILImage(), ]

    # the order is fixed:
    augmentaionFactory = {
        DOWNSAMPLE: Downsample(aug_config[DOWNSAMPLE]) if aug_config[DOWNSAMPLE] else None,
        LIGHTNESS_ADJUST: RandomLightnessAdjustment(),
        CONTRAST_ADJUST: RandomContrastAdjustment(),
        CROP: transforms.RandomCrop(aug_config[CROP]) if aug_config[CROP] else None,
        HORIZON_FLIP: transforms.RandomHorizontalFlip(),
        VERTICAL_FLIP: transforms.RandomVerticalFlip(),
    }

    for k, v in augmentaionFactory.items():
        if aug_config[k]:
            aug_list.append(v)

    aug_list.append(transforms.ToTensor())
    console.log('Dataset augmentation:')
    console.log(aug_list)
    return transforms.Compose(aug_list)


if __name__ == '__main__':
    pass
