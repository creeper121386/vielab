'''
Running multiple models on a same dataset.
'''

from PIL import Image
from globalenv import *

NAME = sys.argv[1]
RES_DIRS = [
    Path('/data1/why/data/train_data'),
    Path('/data1/why/vielab/train_log/0dce/0dce:001@0dcedata/test_result/last.ckpt@0dce-train'),
    Path('/data1/why/vielab/train_log/sshdrnet/sshdrnet:008@0dcedata/test_result/last.ckpt@0dce-train'),
]

console.log(RES_DIRS)


def get_len(folder):
    return len(list(folder.glob('*')))


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


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

    dir_files = [list(x.glob('*')) for x in RES_DIRS]
    for i in range(num0):
        img_list = []
        for j, fpaths in enumerate(dir_files):
            # title = str(RES_DIRS[j])
            # N = 40
            # title = '\n'.join(title[i:i + N] for i in range(0, len(title), N))
            img_list.append(Image.open(fpaths[i]))

        img0 = img_list[0]
        for x in img_list[1:]:
            img0 = get_concat_h(img0, x)

        save_path = save_dir / f'{i}.png'
        img0.save(save_path)
        print(f'[ {i} ] Compare: {RES_DIRS}, Save to: {save_path}')


if __name__ == '__main__':
    main()
    ...
