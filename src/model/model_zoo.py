# Models-mapping
from globalenv import *

from .deeplpf import DeepLpfLitModel
from .ia3dlut import IA3DLUTLitModel

# from .ia3dlut import
MODEL_ZOO = {
    DEEP_LPF: DeepLpfLitModel,
    IA3DLUT: IA3DLUTLitModel,
}
