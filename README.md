# vielab

vielab means Video / Image Enhancement Lab, containing multiple kinds of deep learning based enhancement methods.

methods list:

- [ ] DeepLPF
- [ ] 3D-LUT

## Config

The project use Hydra to manage config files. Config files are located in `./src/config`. `./src/config/config.yaml` defines sub-config groups:

```yaml
defaults: 
  - data: full
  - loss: useall
  - aug: crop512
  - filters: useall
  - model: deeplpf
```

which 