'''
Put read-only global variables here.
'''
from rich.console import Console
console = Console()

# config.yml
PREDICT_ILLUMINATION = 'predict_illumination'
CHECKPOINT_FILEPATH = 'checkpoint_filepath'
VALID_EVERY = 'valid_every'
GT_DIRPATH = 'GT_dirpath'
TRANSFORMS = 'transforms'
NUM_EPOCH = 'num_epoch'
LOG_EVERY = 'log_every'
EXPNAME = 'expname'
LOSSES = 'losses'
GPU = 'gpu'

# config::losses
LOCAL_SMOOTHNESS_LOSS = 'ltvLoss'
COS_SIMILARITY = 'CosSimilarity'
SSIM_LOSS = 'ssimLoss'
L1_LOSS = 'L1Loss'
STRING_FALSE= 'False'

SAVE_MODEL_EVERY = 'save_model_every'
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
INPUT_DIRPATH = 'input_dirpath'

# config::transforms
VERTICAL_FLIP = 'vertical_flip'
HORIZON_FLIP = 'horizon_flip'
RESIZE = 'resize'
CROP = 'crop'

# model.py
INPUT = 'input'
OUTPUT = 'output'

# data.py
INPUT_IMG = 'input_img'
OUTPUT_IMG = 'output_img'
NAME = 'name'

# config mode
TRAIN = 'train'
TEST = 'test'

# file / dir name
IMAGES = 'images'
MODELNAME = 'deep_lpf'


trainNecessaryFields = [
    EXPNAME,
    NUM_EPOCH,
    GPU,
    VALID_EVERY,
    LOG_EVERY,
    SAVE_MODEL_EVERY,
    CHECKPOINT_FILEPATH,
    GT_DIRPATH,
    INPUT_DIRPATH,
    TRANSFORMS,
    PREDICT_ILLUMINATION,
    LOSSES
]

testNecessaryFields = [
    EXPNAME,
    GPU,
    CHECKPOINT_FILEPATH,
    GT_DIRPATH,
    INPUT_DIRPATH,
    TRANSFORMS,
    PREDICT_ILLUMINATION
]