# Models-mapping
from globalenv import *

from model.deeplpf import DeepLpfLitModel
from model.ia3dlut import IA3DLUTLitModel
from model.zerodce import ZeroDCELitModel
from model.hdrnet import HDRnetLitModel

# from .ia3dlut import
MODEL_ZOO = {
    DEEP_LPF: DeepLpfLitModel,
    IA3DLUT: IA3DLUTLitModel,
    ZERODCE: ZeroDCELitModel,
    HDRNET:HDRnetLitModel,
}

# extra required arguments for each model:
RUNTIME_NECESSARY_ARGUMENTS = {
    DEEP_LPF: [
        MODELNAME,
        PREDICT_ILLUMINATION,
        LOSS,
        FILTERS
    ],

    IA3DLUT: [
        MODELNAME,
        MODE,
        COLOR_SPACE,
        BETA1,
        BETA2,
        LAMBDA_SMOOTH,
        LAMBDA_MONOTONICITY,
        LUT_FILEPATH
    ],

    ZERODCE: [
        MODELNAME,
        PREDICT_ILLUMINATION,
        WEIGHT_DECAY,
        GRAD_CLIP_NORM,
        TEST_PTH
    ],

    HDRNET: [
        MODELNAME,
        LOSS,
        PREDICT_ILLUMINATION,
        LUMA_BINS,
        CHANNEL_MULTIPLIER,
        SPATIAL_BIN,
        BATCH_NORM,
        LOW_RESOLUTION,
    ]
}
