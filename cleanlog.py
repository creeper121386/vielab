import os
import os.path as osp
import shutil


def clean(logdir):
    for dirname in os.listdir(logdir):
        dirpath = osp.join(logdir, dirname)

        if not osp.isdir(dirpath):
            continue

        imgpath = os.path.join(dirpath, 'images')
        
        if len(os.listdir(dirpath)) < 3 or len(os.listdir(imgpath)) < 1:
            print('clean:', dirpath)
            shutil.rmtree(dirpath)


clean('train_log')
clean('test_log')
