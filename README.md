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
- prompt-toolkit

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
checkpoint_path: false # false for do not load runtime. Necessary when testing.
```

which is shared by all the models.

Config groups contains `ds`, `aug` and `runtime`. Each config group is defined in directory `./src/config/<group_name>`.
For example, if you want to create a new config file for group `ds`, just create `./src/config/ds/local-debug.yaml` and
fill it. In such a yaml file, you need to add an extra line at the beginning:

```
# @package _group_
```

which means using the group name as package when including this yaml file.

If you have completed the `yaml` file, you need 4 steps to run a model:

- choose your dataset by passing command line arguments `ds=<dataset_name>`
- choose your model and runtime config by passing command line arguments `runtime=<model_name>.<runtime_config>`

[comment]: <> (  > Pay attention to the leading `+` of the arguments. `+` is necessary to **add the argument to the config dict** when this argument is not added to the `defaults` list in `config.yaml`.)

- (not necessary) modify extra general configs if you want, for example: `aug=crop512`
- (not necessary) modify any config option if you want, for example: `aug.resize=true gpu=[1,2]`. Note that if the
  arguments conflict, the previous argument will be overwritten by the later one.

Finally, your command looks like:

```shell
python train.py ds=my_data runtime=deeplpf.config1 aug=resize runtime.loss.ltv=0.001 gpu=2 name=demo1
```

You could run this command in bash or zsh, but a better way is to run in `vash`: a simple "shell" only for this project.
For more details, see the [Running in vash](#running) part.

### What about testing and evaluation?

Assume that you have created config files for training a model. I suggest you only change those necessary parameters
when testing/evaluating the same model (like `name` and `checkpoint_path`). Just left other parameters their default
value and don't care about them. indeed, when you do so, some parameters for training like `num_epoch`, `valid_every`
will also be loaded at the same time, but it's OK because `test.py` and `eval.py` will ignore them.

If you want to define some extra necessary parameters for testing/evaluation other than training parameters, you could
create a new `yaml` in `runtime` group.

### Use logger

The project uses `comet_ml` as experiment logger. Before running, make sure you have an account in comet_ml and set the
environment variables as follows:

```shell
export COMET_API_KEY=<your_api_key>
export COMET_WORKSPACE=<your_workspace>
```

If you don't have `comet_ml` account, you could use other loggers supported by `pytorch-lightning` if you want.

### Use GPU

Argument `gpu` can be an integer which means number of gpus to train on, or a list which GPUs to train on.

Examples:

```shell
# use 2 gpus
python train.py [OTHER_ARGS...] gpu=2     

# don't use gpu
python train.py [OTHER_ARGS...] gpu=0     

# use gpu1, gpu2 and gpu3 (Note that the quotes are necessary!)
python train.py [OTHER_ARGS...] gpu='[1,2,3]'   
```

### Reference

If you are still confused about the config process of the project, I recommend you to
read [Hydra documentation](https://hydra.cc/docs/intro). Trust me, although `Hydra` looks more complicated, it's much
better than `argparse` in deep learning project.

## <a name="running"></a> Running in `vash`

In the `Config your model`, I have told you how to train or test a model using `Hydra`. However, a better way is to run
the command in `vash`.

Because arguments auto-completion and path completion is not supported by hydra, it is so annoying to enter extremely
long arguments in shell. So I implement `vash` with `prompt_toolkit` to solve the problem, which supports path
completion, argument completion, input history suggestion, and even multi-level argument search! Try and enjoy it by
running:

```shell
# train a model:
./vash train

# test or evaluate a model:
./vash test
```

then you will enter a "shell" to input all the arguments with amazing auto-completion. Path completion is supported for
arguments `ds.GT`, `ds.input` and `checkpoint_path`. See the demo gif:

![demo gif](figures/output.gif)

You can also press `up` and `down` to toggle in history commands or press `Ctrl + r` to search in history like in bash.
