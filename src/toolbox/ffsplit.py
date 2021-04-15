import os
import sys

from rich.console import Console

'''
Split video to image sequences
Usage: python ffsplit.py <video_dir_path> <output_dir_path> <resize_factor>
or:    python ffsplit.py <video_path> <output_dir_path> <resize_factor>
'''

c = Console()
assert len(sys.argv) >= 3

src = sys.argv[1]
dstdir = sys.argv[2]
resize = sys.argv[3]


def splitOne(fpath, dst):
    if not os.path.exists(dst):
        os.system('mkdir {}'.format(dst))
    fname = os.path.basename(fpath)

    os.system(f'mkdir {os.path.join(dst, fname)}')

    dstglob = os.path.join(dst, fname, f'{fname}-%d.png')
    cmd = f'ffmpeg -i {fpath} -vf scale=iw*{resize}:ih*{resize}:flags=bicubic {dstglob}'
    c.log('Running cmd: ' + cmd)
    os.system(cmd)


if os.path.isdir(src):
    for x in os.listdir(src):
        splitOne(os.path.join(src, x), dstdir)
elif os.path.exists(src):
    splitOne(src, dstdir)
else:
    c.log('Source video file not found.')
