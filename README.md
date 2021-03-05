# vielab

vielab means Video / Image Enhancement Lab, containing multiple kinds of deep learning based enhancement methods:

- [x] DeepLPF
- [ ] 3D-LUT

## Config

The project use `Hydra` to manage config files. Config files are located in `./src/config`. `./src/config/config.yaml`
defines sub-config groups, e.g:

```yaml
defaults:
  - ds: full
  - loss: useall
  - aug: crop512
  - model: deeplpf
```

Config groups are defined in `./src/config/<group_name>`. For example, you can create a new config group or create a new
config file `./src/config/ds/local-debug.yaml` for group `ds`. In such a yaml file, you need to add an extra line at the
beginning:

```
# @package _group_
```

which defines the package name when including this yaml file.

The entire config process contains 3 steps:

- choose your dataset: `ds=<dataset_name>`
- choose your model and runtime config: `runtime=<model_name>.<runtime_config>`
- modify extra general configs if you want, for example: `aug=crop512`