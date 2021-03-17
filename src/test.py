# -*- coding: utf-8 -*-
import hydra
import torch
from data import ImagesDataset
from globalenv import *
from pytorch_lightning import Trainer
from util import checkConfig, configLogging, parseAugmentation


@hydra.main(config_path='config', config_name="config")
def main(opt):
    opt = checkConfig(opt, TEST)
    opt[LOG_DIRPATH], opt[IMG_DIRPATH] = configLogging(TEST, opt)

    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    modelname = opt[RUNTIME][MODELNAME]
    if modelname not in MODEL_ZOO:
        raise RuntimeError(f'ERR: Model {modelname} not found. Please change the argument `runtime.modelname`')
    ModelClass = MODEL_ZOO[modelname]

    assert opt[CHECKPOINT_PATH]
    model = ModelClass.load_from_checkpoint(opt[CHECKPOINT_PATH], opt=opt)
    console.log(f'Loading model from: {opt[CHECKPOINT_PATH]}')

    transform = parseAugmentation(opt)
    ds = ImagesDataset(opt, data_dict=None, ds_type=DATA, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=opt[DATALOADER_NUM_WORKER]
    )
    trainer = Trainer(gpus=opt[GPU], distributed_backend='dp')

    trainer.test(model, dataloader)


if __name__ == "__main__":
    main()
