[comment]: <> (# vielab)

![logo](./figures/logo.png)

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/creeper121386/vielab?color=yellow&style=for-the-badge)](https://github.com/creeper121386/vielab/stargazers) 
 &nbsp; [![GitHub issues](https://img.shields.io/github/issues/creeper121386/vielab?style=for-the-badge)](https://github.com/creeper121386/vielab/issues)
&nbsp; [![GitHub forks](https://img.shields.io/github/forks/creeper121386/vielab?style=for-the-badge)](https://github.com/creeper121386/vielab/network)
&nbsp; ![GitHub repo size](https://img.shields.io/github/repo-size/creeper121386/vielab?color=red&style=for-the-badge)

![](https://img.shields.io/badge/python-3.8-green?style=for-the-badge)
&nbsp; ![](https://img.shields.io/badge/pytorch-1.5-blueviolet?style=for-the-badge)
&nbsp; ![](https://img.shields.io/badge/pytorch--lightning-1.2.2-blueviolet?style=for-the-badge)

</div>
</br>

`vielab`(Video / Image Enhancement Lab) is a PyTorch codebase for computer vision researchers, encapsulating the implementation of multiple deep learning based video and image enhancement methods, sharing the same training and testing pipeline. Supported methods:


|Model|Source|Paper | Multi-GPU Training
|:---:|:---:|:---: | :---: 
|deeplpf|[github repo](https://github.com/sjmoran/DeepLPF)|[DeepLPF: Deep Local Parametric Filters for Image Enhancement](https://arxiv.org/abs/2003.13985) | ❎
|ia3dlut|[github repo](https://github.com/HuiZeng/Image-Adaptive-3DLUT)|[Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time](https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf) | ❎
|zerodce| [github repo](https://github.com/Li-Chongyi/Zero-DCE) | [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) | ?
|unet| - | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597) | ?
|hdrnet| [github repo (unofficial)](https://github.com/creotiv/hdrnet-pytorch) | [Deep Bilateral Learning for Real-Time Image Enhancements](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf) | ✅
|deepupe<sup>1<sup/>| [github repo](https://github.com/dvlab-research/DeepUPE) | [Underexposed Photo Enhancement Using Deep Illumination Estimation](https://drive.google.com/file/d/1CCd0NVEy0yM2ulcrx44B1bRPDmyrgNYH/view?usp=sharing) | ✅


<div style="color: grey; ">
1. The implementation of DeepUPE and HDRnet share the same code because the only difference between them is that DeepUPE has an extra illumination-map-prediction layer (controlled by `runtime.predict_illumination=true` in config).
</div>

[comment]: <> (The framework of `vielab` is applicable to **ANY** model whose input and output are both images, like de-noising, HDR, super resolution, and other kinds of enhancement.)

The project is developed all on my own and for personal use. Feel free to pull request and open issues! 

## Environment

- Pytorch 1.5
- cuda 10.2
- cudnn 7
- pytorch-lightning 1.2.2
- prompt-toolkit
- wandb

## Config

### Training

The project use `Hydra` to manage config files. The main config files is `./src/config/config.yaml`, which contains two parts: 
- **General configs** for all models and experiments, like `name, num_epoch, checkpoint_path`.
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

- **Config groups** for each individual experiment: 
  - `ds`(dataset config),
  - `aug`(data augmentation config)
  - `runtime`(model config). 
    
  Config group files are in `./src/config/<group_name>`. For example, to add a new config file in group `ds`, just create `./src/config/ds/new_dataset.yaml`. In such a yaml file, you need to add an extra line at the beginning:
  
  ```
  # @package _group_
  ```

  which means using the group name as package name when including this yaml file.
  
To train a model, you need to:

- choose the dataset by passing command line arguments `ds=<dataset_name>`
- choose the model (runtime config) by passing `runtime=<runtime_name>`

- Change any configs if you want, for example: `aug=crop512` or `aug.resize=true gpu=[1,2]`. 
  
P.S. If the arguments are conflict, the previous argument will be overwritten by the later one. An example:

```shell
python train.py ds=my_data runtime=deeplpf.config1 aug=resize runtime.loss.ltv=0.001 gpu=2 name=demo1
```

You could run the command in `bash` or `zsh`, but a better choice is `vsh`: a simple "shell" for `vielab`. For more details, see [Running in vsh](#running).

### Test and evaluation

If you have created config files for training a model, you only need to change those necessary parameters (like `name`, `runtime`, `ds` and `checkpoint_path`) to test/evaluate the same model. Just left other training parameters default (like `num_epoch`, `valid_every`), `test.py` will ignore them. An example:

```shell
ipython test.py  checkpoint_path=../train_log/hdrnet/hdrnet-adobe5k-010/last.ckpt ds=adobe5k.train valid_ds=adobe5k.valid runtime=hdrnet.default runtime.predict_illumination=true aug=none
```

## logging

### Use logger

The project uses `wandb` as logger. Before running, make sure you have an account in `wandb`.

### `LightningModule.log`

Call `BaseModel.log(name, value, on_step, on_epoch)` to auto-log the metrics to the `wandb` logger. In pytorch-lightning doc:

> Depending on where log is called from, Lightning auto-determines the correct logging mode for you.

In details, if you call `self.log` in `training_step`, the default setting is
`on_step=True, on_epoch=False` and if you call `self.log` in `validation_step`, then the default setting is`on_step=False, on_epoch=True`.

So just call `self.log(name, value)` and leaving its arguments default.

### log images

Create your model from `model.BaseModel` and use following methods to log your images:

- `self.save_one_img_of_batch(batch, dirpath, fname)`: save `batch[0]` to `dirapth / fname`.
- `log_images_dict(mode, input_fname, img_batch_dict)`: save images in `img_batch_dict` to filesystem and `wandb` logger.

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

Argument `gpu` is an integer which means number of gpus to train on, or a list which GPUs to train on.

Examples:

```shell
# use 2 any gpus
python train.py [OTHER_ARGS...] gpu=2     

# don't use gpu
python train.py [OTHER_ARGS...] gpu=0     

# use gpu1, gpu2 and gpu3
python train.py [OTHER_ARGS...] gpu=[1,2,3]
```


## <a name="running"></a> Running in `vsh`

`Hydra` is wonderful but running `Hydra` app in `bash` is annoying cause auto-completion is not supported. So I develop `vsh` to solve the problem.

`vsh` supports path completion, argument completion, input history suggestion, and even dynamic `yaml` argument completion! Try and enjoy it! Usage:

```shell
# train a model:
./vsh train

# test or evaluate a model:
./vsh test

# export onnx model:
./vsh onnx
```

Path completion is supported for arguments `ds.GT`, `ds.input` and `checkpoint_path`. Demo gif:

![demo gif](figures/output.gif)

Press `up` and `down` to toggle in history commands and press `Ctrl + r` to search in history like bash.

## Implement a new model

There are 3 steps to add the source code of a `new_model` to `vielab`:

- Implement the class `NewModel` in `src/model/new_model.py`, inheriting `BaseModel`.
- Add a new line to import `NewModel` in function  `parse_model_class` in `src/model/model_zoo.py`
- Create runtime config file in `config/runtime/new_model.default.yaml`.
- If any new `str` constant is used, declare it in `globalenv.py`.