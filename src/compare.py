'''
Running multiple models on a same dataset.
'''
import os
import re
from pathlib import Path

from PIL import Image
from globalenv import TEST_RESULT_DIRNAME, TRAIN_LOG_DIRNAME, console
from matplotlib import pyplot as plt

DATASET = '0dcedata.train'
DATASET_CONFIG = f'ds={DATASET}'
AUG_CONFIG = 'aug=none'
RUNTIME_ARGS = [
    'checkpoint_path=../train_log/0dce/0dce:001@0dcedata/last.ckpt \
    runtime=0dce.default',
    'checkpoint_path=../train_log/sshdrnet/sshdrnet:008@0dcedata/last.ckpt ds=0dcedata.train runtime=sshdrnet.default'
]

RES_DIRS = [
    sorted(
        (
                Path(
                    re.findall(f'\.\./{TRAIN_LOG_DIRNAME}.*ckpt', x)[0]
                ).parent / TEST_RESULT_DIRNAME
        ).glob(f'*'), key=os.path.getmtime
    )[0]
    for x in RUNTIME_ARGS
]

console.log(RES_DIRS)


def get_len(folder):
    return len(list(folder.glob('*')))


def main():
    # Running
    # for arg in RUNTIME_ARGS:
    #     cmd = f'python test.py {DATASET_CONFIG} {AUG_CONFIG} {arg}'
    #     print('[ RUN ]', cmd)
    #     os.system(cmd)

    # visiualize result
    save_dir = Path(f'../compare_log/{DATASET}')
    if save_dir.exists():
        print('[ ERR ] result dir exists. Please use a new one.')
    else:
        save_dir.mkdir(parents=True)

    num0 = get_len(RES_DIRS[0])
    for dirpath in RES_DIRS:
        assert num0 == get_len(dirpath)

    dir_files = [list(x.glob('*')) for x in RES_DIRS]
    for i in range(num0):
        fig = plt.figure()
        for j, fpaths in enumerate(dir_files):
            ax = fig.add_subplot(1, len(RES_DIRS), j+1)
            ax.set_title(re.findall(f'\.\./{TRAIN_LOG_DIRNAME}/.*?/', str(RES_DIRS[j]))[0])
            plt.imshow(Image.open(fpaths[i]))

        save_path = save_dir / f'{i}.png'
        plt.savefig(save_path, dpi = 500)
        print(f'[ {i} ] Compare: {RES_DIRS}, Save to: {save_path}')


if __name__ == '__main__':
    main()
    ...
