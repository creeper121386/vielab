# vielab

vielab means Video / Image Enhancement Lab, containing multiple kinds of deep learning based enhancement methods.

methods list:

- [ ] DeepLPF
- [ ] 3D-LUT

## Config

The project use Hydra to manage config files. Config files are located in `./src/config`. `./src/config/config.yaml` defines sub-config groups:

```yaml
defaults: 
  - ds: full
  - loss: useall
  - aug: crop512
  - filters: useall
  - model: deeplpf
```
Each config groups are defined in `./src/config/<group_name>`. For example, you can define a new config file `./src/config/ds/local-debug.yaml` for group `ds`. In such a yaml file, you need to add an extra line at the beginning:

```
# @package _group_
```
which defines the package name when including this yaml file.