# vielab

`vielab` means Video / Image Enhancement Lab, containing multiple kinds of deep learning based video and image
enhancement methods:

[comment]: <> (- [x] DeepLPF: [src]&#40;https://github.com/sjmoran/DeepLPF&#41; | [paper]&#40;https://arxiv.org/abs/2003.13985&#41;)

[comment]: <> (- [ ] 3D-LUT: [src]&#40;https://github.com/HuiZeng/Image-Adaptive-3DLUT&#41; | [paper]&#40;https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf&#41;)

|Model|Source|Paper
|---|---|---
|DeepLPF|https://github.com/sjmoran/DeepLPF|https://arxiv.org/abs/2003.13985
|3D-LUT|https://github.com/HuiZeng/Image-Adaptive-3DLUT|https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf

The code architecture of this project is applicable to any model whose input and output are images, like de-noising,
HDR, super resolution, and other kinds of enhancement.

## Environment

- Pytorch 1.5
- cuda 10.2
- cudnn 7
- pytorch-lightning 1.2.2

## Config your model

### Training

The project use `Hydra` to manage config files. Config files are located in `./src/config`. `./src/config/config.yaml`
contains two parts: general configs for all experiments and config groups for each experiment. General config looks
like:

```yaml
name: default_name    # name of experiment
num_epoch: 1000
gpu: 1
valid_every: 10
log_every: 100
save_model_every: 20
model_path: false # false for do not load runtime. Necessary when testing.
```

which is shared by all the models.

Config groups contains `ds`, `aug` and `runtime`. Each config group is defined in directory `./src/config/<group_name>`.
For example, if you want to create a new config file for group `ds`, just create `./src/config/ds/local-debug.yaml` and
fill it. In such a yaml file, you need to add an extra line at the beginning:

```
# @package _group_
```

which means using the group name as package when including this yaml file.

If you have finished modified the `yaml` file, you need 4 steps to run a model:

- choose your dataset by passing command line arguments `+ds=<dataset_name>`
- choose your model and runtime config by passing command line arguments `+runtime=<model_name>.<runtime_config>`

  > Pay attention to the leading `+` of the arguments. `+` is necessary to **add the argument to the config dict** when this argument is not added to the `defaults` list in `config.yaml`.

- (not necessary) modify extra general configs if you want, for example: `aug=crop512`
- (not necessary) modify any config option if you want, for example: `aug.resize=true`. Note that if the arguments
  conflict, the previous argument will be overwritten by the later one.

Finally, your command looks like:

```shell
python train.py +ds=my_data +runtime=deeplpf.config1 aug=resize runtime.loss.ltv=0.001 gpu=2 name=demo1
```

### What about testing and evaluation?

Assume that you have created config files for training a model. I suggest you only change those necessary parameters
when testing/evaluating the same model. (like `name` and `model_path`) Just left other parameters their default value
and don't care about them. indeed, when you do so, some parameters for training like `num_epoch`, `valid_every` will
also be loaded at the same time, but it's OK cause `test.py` and `eval.py` will ignore them.

If you want to define some extra necessary parameters for testing/evaluation other than training parameters, you could
create a new `yaml` in `runtime` group.

### Use GPU

Argument `gpu` can be an integer which means number of gpus to train on, or a list which GPUs to train on.

Examples:

```shell
python train.py [OTHER_ARGS...] gpu=2     # use 2 gpus
python train.py [OTHER_ARGS...] gpu=0     # don't use gpu
python train.py [OTHER_ARGS...] gpu='[1,2,3]'   # use gpu1, gpu2 and gpu3
```

### Reference

If you are still confused about the config process of the project, I recommend you to
read [Hydra documentation](https://hydra.cc/docs/intro). Trust me, although `Hydra` looks more complicated, it's much
better than `argparse` in deep learning project.