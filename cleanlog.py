import os
import os.path as osp
import shutil


def clean(logdir):
    for dirname in os.listdir(logdir):
        dirpath = osp.join(logdir, dirname)

        if not osp.isdir(dirpath):
            continue

        if len(os.listdir(dirpath)) < 3:
            print('clean:', dirpath)
            shutil.rmtree(dirpath)


clean('train_log')
clean('test_log')
