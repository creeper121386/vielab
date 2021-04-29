# -*- coding: utf-8 -*-
import logging

import hydra
from data_aug import parseAugmentation
from dataset import ImagesDataset
from globalenv import *
from model.model_zoo import parse_model_class


@hydra.main(config_path='config', config_name="config")
def main(config):
    opt = parse_config(config, TRAIN)

    # logging
    console.log('Running config:', opt, log_locals=False)
    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False

    # init model:
    ModelClass = parse_model_class(opt[RUNTIME][MODELNAME])
    if opt[CHECKPOINT_PATH]:
        model = ModelClass.load_from_checkpoint(opt[CHECKPOINT_PATH], opt=opt)
        console.log(f'Loading model from: {opt[CHECKPOINT_PATH]}')
    else:
        model = ModelClass(opt)

    # Loading data:
    transform = parseAugmentation(opt)
    dataset = ImagesDataset(opt, ds_type=DATA, transform=transform)
    console.log('Finish loading data.')
    sample_input_batch = dataset[0][INPUT].unsqueeze(0)

    # only support one input argument in model.forward
    dynamic_ax = {
        'input': {2: 'H', 3: 'W'},
        'output': {2: 'H', 3: 'W'}
    }
    fpath = f"../onnx/{opt[NAME]}.onnx"
    model.to_onnx(
        fpath, sample_input_batch,
        export_params=True,
        verbose=True,
        do_constant_folding=True,
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        opset_version=12,
        # dynamic_axes=dynamic_ax
    )
    console.log(f'[[ DONE ]] ONNX file {fpath} exported.')


if __name__ == "__main__":
    main()
