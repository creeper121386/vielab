import os
import os.path as osp

logdir = 'train_log'
for dirname in os.listdir():
    dirpath = osp.join(logdir, dirname)

    if len(os.listdir(dirpath)) < 2:
        print('clean:', dirpath)
        os.removedirs(dirpath)