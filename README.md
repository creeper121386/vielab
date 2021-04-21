# vielab

`vielab`: Video / Image Enhancement Lab, containing multiple kinds of deep learning based video and image
enhancement methods:

[comment]: <> (- [x] DeepLPF: [src]&#40;https://github.com/sjmoran/DeepLPF&#41; | [paper]&#40;https://arxiv.org/abs/2003.13985&#41;)

[comment]: <> (- [ ] 3D-LUT: [src]&#40;https://github.com/HuiZeng/Image-Adaptive-3DLUT&#41; | [paper]&#40;https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf&#41;)

|Model|Source|Paper
|:---:|:---:|:---:
|deeplpf|[github repo](https://github.com/sjmoran/DeepLPF)|[paper link](https://arxiv.org/abs/2003.13985)
|ia3dlut|[github repo](https://github.com/HuiZeng/Image-Adaptive-3DLUT)|[paper link](https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf)
|zerodce| [github repo](https://github.com/Li-Chongyi/Zero-DCE) | [paper link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) 
|unet| - | - 
|hdrnet| [github repo (unofficial)](https://github.com/creotiv/hdrnet-pytorch) | [paper link](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf)

The code architecture of this project is applicable to **ANY** model whose input and output are images, like de-noising,
HDR, super resolution, and other kinds of enhancement.

## Environment

- Pytorch 1.5
- cuda 10.2
- cudnn 7
- pytorch-lightning 1.2.2
- prompt-toolkit
- wandb

## Config your model

### Training

The project use `Hydra` to manage config files. Config files are located in `./src/config`. `./src/config/config.yaml`
contains two parts: 
- General configs for all experiments, like `name, num_epoch, checkpoint_path`
  General config looks like:
  
  ```yaml
  name: default_name    # name of experiment
  num_epoch: 1000
  gpu: 1
  valid_every: 10
  log_every: 100
  save_model_every: 20
  checkpoint_path: false # false for do not load runtime. Necessary when testing.
  ```
  
  which is shared by all models.
  

- And config groups for each experiment, containing `ds`, `aug` and `runtime`. Config group is defined in directory `./src/config/<group_name>`.
  
  For example, if you want to add a new config file for group `ds`, just create `./src/config/ds/local-debug.yaml` and
  fill it. In such a yaml file, you need to add an extra line at the beginning:
  
  ```
  # @package _group_
  ```

  which means using the group name as package when including this yaml file.
  
You need 4 steps to passing arguments to run a model:

- choose the dataset by passing command line arguments `ds=<dataset_name>`
- choose your model (runtime config) by passing command line arguments `runtime=<runtime_name>`

[comment]: <> (  > Pay attention to the leading `+` of the arguments. `+` is necessary to **add the argument to the config dict** when this argument is not added to the `defaults` list in `config.yaml`.)

- (not necessary) Change extra general configs if you want, for example: `aug=crop512` or `aug.resize=true gpu=[1,2]`. Note that if the
  arguments conflict, the previous argument will be overwritten by the later one.

Finally, your command looks like:

```shell
python train.py ds=my_data runtime=deeplpf.config1 aug=resize runtime.loss.ltv=0.001 gpu=2 name=demo1
```

You could run this command in bash or zsh, but a better way is to run in `vsh`: a simple "shell"  for `vielab`. For more details, see the [Running in vsh](#running) part.

### Test and evaluation

Assume that you have created config files for training a model. You only need to change those necessary parameters
to switch to test/evaluation mode for the same model (like `name`, `runtime`, `ds` and `checkpoint_path`). Just left other parameters their default
value and don't care about them. Indeed, when you do so, some parameters for training like `num_epoch`, `valid_every`
will also be passed, but it's OK because `test.py` will ignore them.

## logging

### Use logger

The project uses `wandb` as experiment logger. Before running, make sure you have an account in `wandb`. At the first running, `wandb` will require you to login the account.

### `LightningModule.log`

When creating new model class in `model/`, you could call `self.log(name, value, on_step, on_epoch)` in
your `LightningModule` class to auto-log the metrics to the `wandb` logger. In pytorch-lightning doc:

> Depending on where log is called from, Lightning auto-determines the correct logging mode for you.

which means if you call `self.log` in `training_step`,
`on_step=True, on_epoch=False` by default. if you call `self.log` in `validation_step`, `on_step=False, on_epoch=True`
by default.

So just call `self.log(name, value)` and leaving the arguments' default values is OK.

### log images

Create your pytorch-lightning module from `model.BaseModel`. Use following methods to log your images:

- `self.save_one_img_of_batch(batch, dirpath, fname)`: save `batch[0]` to disk.
- `log_images_dict(mode, input_fname, img_batch_dict)`: save images in `img_batch_dict` to disk and remote logger.

### Send email

add your smtp server password as env variable `SMTP_PASSWD`:

```shell
export SMTP_PASSWD=<your_password>
```

And modify following parts in `globalenv.py`:

```python
# email info:
FROM_ADDRESS = '344915973@qq.com'
TO_ADDRESS = 'hust.why@qq.com'
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 465
```

The mail will be sent once exception occurred or running finished.

## Use GPU

Argument `gpu` can be an integer which means number of gpus to train on, or a list which GPUs to train on.

Examples:

```shell
# use 2 gpus
python train.py [OTHER_ARGS...] gpu=2     

# don't use gpu
python train.py [OTHER_ARGS...] gpu=0     

# use gpu1, gpu2 and gpu3
python train.py [OTHER_ARGS...] gpu=[1,2,3]   
```

## Reference

If you are still confused about the config process of the project, I recommend you to
read [Hydra documentation](https://hydra.cc/docs/intro). Although `Hydra` looks more complicated, it's much
better than `argparse` in deep learning project.

## <a name="running"></a> Running in `vsh`

`Hydra` is wonderful but running `Hydra` app in `bash` is annoying cause auto-completion is not supported. So I develop `vsh`with `prompt_toolkit` to solve the problem.

`vsh` supports path completion, argument completion, input history suggestion, and even dynamic `yaml` argument completion! Try and enjoy it by
running:

```shell
# train a model:
./vsh train

# test or evaluate a model:
./vsh test
```

Path completion is supported for arguments `ds.GT`, `ds.input` and `checkpoint_path`. See the demo gif:

![demo gif](figures/output.gif)

You could also press `up` and `down` to toggle in history commands or press `Ctrl + r` to search in history like bash.
