'''
Running multiple models on a same dataset.
'''
import ipdb
from PIL import Image
from globalenv import *

NAME = sys.argv[1]
RES_DIRS = [
    '/home/rxwang/why/data/fulldata-0329/train/input',
    '/home/rxwang/why/vielab/train_log/0dce/0dce:001@fulldata0329/test_result/last.ckpt@fulldata',
    '/home/rxwang/why/data/3dlut-fulldata-output',
    '/home/rxwang/why/vielab/train_log/hdrnet/hdrnet:004@fulldata/test_result/last.ckpt@fulldata',

    '/home/rxwang/why/data/fulldata-0329/train/output',

    # Path('/data1/why/data/train_data'),
    # Path('/data1/why/vielab/train_log/0dce/0dce:001@0dcedata/test_result/last.ckpt@0dce-train'),
    # Path('/data1/why/vielab/train_log/sshdrnet/sshdrnet:008@0dcedata/test_result/last.ckpt@0dce-train'),
]

console.log(RES_DIRS)
RES_DIRS = [Path(x) for x in RES_DIRS]


def get_len(folder):
    return len(list(folder.glob('*')))


console.log('Folder file nums:')
console.log([get_len(x) for x in RES_DIRS])


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def fix_imgs_size(img_list):
    heights = [x.size[1] for x in img_list]
    fixed_height = min(heights)
    return [x.resize([int(x.size[0] * fixed_height / x.size[1]), fixed_height]) for x in img_list]
    # ipdb.set_trace()


def main():
    # visiualize result
    save_dir = Path(f'../compare_log/{NAME}')
    if save_dir.exists():
        print('[ WARN ] result dir exists. The existing dir will be overwritten. Continue? (ENTER / Ctrl-c)')
        input()
    else:
        save_dir.mkdir(parents=True)

    num0 = get_len(RES_DIRS[0])
    for dirpath in RES_DIRS:
        assert num0 == get_len(dirpath)

    dir_files = [list(x.glob('*')) for x in RES_DIRS]       # 2d-list
    for i in range(num0):
        img_list = []
        log_fnames = []
        for j, fpaths in enumerate(dir_files):
            # for each folder, add ith image:

            # title = str(RES_DIRS[j])
            # N = 40
            # title = '\n'.join(title[i:i + N] for i in range(0, len(title), N))
            img_list.append(Image.open(fpaths[i]))
            log_fnames.append(str(fpaths[i]))

        img_list = fix_imgs_size(img_list)

        img0 = img_list[0]
        for x in img_list[1:]:
            img0 = get_concat_h(img0, x)

        save_path = save_dir / f'{i}.png'
        img0.save(save_path)
        print(f'[ {i} ] Compare: {log_fnames}, Save to: {save_path}')


if __name__ == '__main__':
    main()
    ...
