"""
Put read-only global variables here.
! DO NOT EDIT THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING
"""
import sys
from pathlib import Path

from rich.console import Console
from rich.traceback import install

install()
console = Console()
SRC_PATH = Path(__file__).absolute().parent
print(f'[ ENV ] add directory {str(SRC_PATH)} to path.')
sys.path.append(str(SRC_PATH))

# from torch.cuda import is_available as cuda_available
# CUDA_AVAILABLE = cuda_available()

########################################
### The following part CAN be edited ###
# Constants
TEST_RESULT_DIRNAME = 'test_result'
TRAIN_LOG_DIRNAME = 'train_log'
CONFIG_DIR = 'config'
CONFIG_FILEPATH = 'config/config.yaml'
METRICS_LOG_DIRPATH = Path(SRC_PATH).parent / 'metrics_log'
OPT_FILENAME = 'CONFIG'
LOG_FILENAME = 'run.log'
LOG_TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
INPUT = 'input'
OUTPUT = 'output'
GT = 'GT'
STRING_FALSE = 'False'
SKIP_FLAG = 'q'
DEFAULTS = 'defaults'
HYDRA = 'hydra'

# fields in dataset.py (result of dataset[i])
FPATH = 'fpath'

# config.yml
DEBUG = 'debug'
CHECKPOINT_PATH = 'checkpoint_path'
LOG_DIRPATH = 'log_dirpath'
IMG_DIRPATH = 'img_dirpath'
DATALOADER_NUM_WORKER = 'dataloader_num_worker'
VALID_EVERY = 'valid_every'
AUGMENTATION = 'aug'
NUM_EPOCH = 'num_epoch'
LOG_EVERY = 'log_every'
NAME = 'name'
LOSS = 'loss'
DATA = 'ds'
VALID_DATA = 'valid_ds'
GPU = 'gpu'
RUNTIME = 'runtime'
MODELNAME = 'modelname'
BATCHSIZE = 'batchsize'
VALID_BATCHSIZE = 'valid_batchsize'
LR = 'lr'
CHECKPOINT_MONITOR = 'checkpoint_monitor'
COMMENT = 'comment'

# config.loss
LTV_LOSS = 'ltv'
COS_LOSS = 'cos'
SSIM_LOSS = 'ssimLoss'
L1_LOSS = 'L1Loss'
COLOR_LOSS = 'l_color'
SPATIAL_LOSS = 'l_spa'
EXPOSURE_LOSS = 'l_exp'

# metrics
PSNR = 'psnr'
SSIM = 'ssim'

# config.aug
VERTICAL_FLIP = 'v-flip'
HORIZON_FLIP = 'h-flip'
DOWNSAMPLE = 'downsample'
CROP = 'crop'
LIGHTNESS_ADJUST = 'lightness_adjust'
CONTRAST_ADJUST = 'contrast_adjust'

# config.runtime.modelname
DEEP_LPF = 'deeplpf'
IA3DLUT = '3dlut'
ZERODCE = '0dce'
HDRNET = 'hdrnet'
SELF_SUPERVISED_SHDRNET = 'sshdrnet'

# config.runtime.deeplpf.*
PREDICT_ILLUMINATION = 'predict_illumination'
FILTERS = 'filters'
USE_GRADUATED_FILTER = 'g'
USE_ELLIPTICAL_FILTER = 'e'

# config.runtime.3dlut.*
MODE = 'mode'
COLOR_SPACE = 'color_space'
BETA1 = 'beta1'
BETA2 = 'beta2'
LAMBDA_SMOOTH = 'lambda_smooth'
LAMBDA_MONOTONICITY = 'lambda_monotonicity'
MSE = 'mse'
TV_CONS = 'tv_cons'
MN_CONS = 'mv_cons'
WEIGHTS_NORM = 'wnorm'
LUT_FILEPATH = 'lut_filepath'
TEST_PTH = 'test_pth'

# config.runtime.0zero.*
WEIGHT_DECAY = 'weight_decay'
GRAD_CLIP_NORM = 'grad_clip_norm'

# config.runtime.hdrnet.*
LUMA_BINS = 'luma_bins'
CHANNEL_MULTIPLIER = 'channel_multiplier'
SPATIAL_BIN = 'spatial_bin'
BATCH_NORM = 'batch_norm'
NET_INPUT_SIZE = 'net_input_size'
LOW_RESOLUTION = 'low_resolution'
ONNX_EXPORTING_MODE = 'onnx_exporting_mode'
SELF_SUPERVISED = 'self_supervised'
GUIDEMAP = 'guidemap'

# config.runtime.sshdrnet.*
USE_HSV = 'use_hsv'

# running mode
TRAIN = 'train'
TEST = 'test'
VALID = 'valid'
ONNX = 'onnx'

# file / dir name
IMAGES = 'images'

# required arguments in any condition:
GENERAL_NECESSARY_ARGUMENTS = [
    NAME,
    RUNTIME,
    GPU,
    DATA,
]

# extra required arguments for all models when training:
TRAIN_NECESSARY_ARGUMENTS = [
    NUM_EPOCH,
    VALID_EVERY,
    LOG_EVERY,
    # SAVE_MODEL_EVERY,
    AUGMENTATION,
    VALID_BATCHSIZE
]

# extra required arguments for all models when testing/evaluating:
TEST_NECESSARY_ARGUMENTS = [
    CHECKPOINT_PATH,
]

# email info:
FROM_ADDRESS = '344915973@qq.com'
TO_ADDRESS = 'hust.why@qq.com'
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 465

#############################################
#### The following part CAN NOT be edited ###

ARGUMENTS_MISSING_ERRS = {}
for ls in [GENERAL_NECESSARY_ARGUMENTS, TRAIN_NECESSARY_ARGUMENTS, TEST_NECESSARY_ARGUMENTS]:
    for x in ls:
        ARGUMENTS_MISSING_ERRS[x] = f'[ ERR ] Config missing argument: {x}'
