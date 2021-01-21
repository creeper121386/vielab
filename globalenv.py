'''
Put read-only global varibles here.
'''

NUM_EPOCH = 'num_epoch'
GPU = 'gpu'
VALID_EVERY = 'valid_every'
SAVE_MODEL_EVERY = 'save_model_every'
CHECKPOINT_FILEPATH = 'checkpoint_filepath'
GT_DIRPATH = 'GT_dirpath'
INPUT_DIRPATH = 'input_dirpath'
TRANSFORMS = 'transforms'
RESIZE = 'resize'
CROP = 'crop'
HORIZON_FLIP = 'horizon_flip'
VERTICAL_FLIP = 'vertical_flip'
EXPNAME = 'expname'


necessaryFields = [
    NUM_EPOCH,
    GPU,
    VALID_EVERY,
    SAVE_MODEL_EVERY,
    CHECKPOINT_FILEPATH,
    GT_DIRPATH,
    INPUT_DIRPATH,
    TRANSFORMS
]

# # outdoor - over exposure
# GT_dirpath = '/data1/rxwang/outdoor/overexpose_gt'
# input_dirpath = '/data1/rxwang/outdoor/overexpose_input'

# # outdoor - under exposure
# GT_dirpath = '/data1/rxwang/outdoor/underexpose_input'
# input_dirpath = '/data1/rxwang/outdoor/underexpose_input'
