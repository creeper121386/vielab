# Models-mapping
from globalenv import *


def parse_model_class(modelname):
    if modelname == DEEP_LPF:
        from model.deeplpf import DeepLpfLitModel as ModelClass

    elif modelname == IA3DLUT:
        from model.ia3dlut import IA3DLUTLitModel as ModelClass

    elif modelname == ZERODCE:
        from model.zerodce import ZeroDCELitModel as ModelClass

    elif modelname == HDRNET:
        from model.hdrnet import HDRnetLitModel as ModelClass

    elif modelname == SELF_SUPERVISED_SHDRNET:
        from model.sshdrnet import HDRnetLitModel as ModelClass

    else:
        raise NotImplementedError(f'[ ERR ] Unknown modelname: {mode}')

    return ModelClass
