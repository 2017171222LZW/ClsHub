# [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
**Declare: This source code is based on ConvNeXt.**

Official PyTorch implementation of **ConvNeXt**, from the following paper:

[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545). CVPR 2022.\
[Zhuang Liu](https://liuzhuang13.github.io), [Hanzi Mao](https://hanzimao.me/), [Chao-Yuan Wu](https://chaoyuan.org/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) and [Saining Xie](https://sainingxie.com)\
Facebook AI Research, UC Berkeley\
[[`arXiv`](https://arxiv.org/abs/2201.03545)][[`video`](https://www.youtube.com/watch?v=QzCjXqFnWPE)]

--- 

## Catalog
- [x] ConvNeXt Training Code  
- [x] ConvNeXt Pre-training Code  
- [x] ConvNeXt Fine-tuning Code



<!-- ✅ ⬜️  -->
<!-- 
## Results and Pre-trained Models
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.1 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| ConvNeXt-S | 224x224 | 83.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 83.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| ConvNeXt-B | 384x384 | 85.1 | 89M | 45.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth) |
| ConvNeXt-L | 224x224 | 84.3 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| ConvNeXt-L | 384x384 | 85.5 | 198M | 101.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |

### ImageNet-22K trained models

| name | resolution |acc@1 | #params | FLOPs | 22k model | 1k model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.9 | 29M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth)
| ConvNeXt-T | 384x384 | 84.1 | 29M | 13.1G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth)
| ConvNeXt-S | 224x224 | 84.6 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth)
| ConvNeXt-S | 384x384 | 85.8 | 50M | 25.5G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth)
| ConvNeXt-B | 224x224 | 85.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth)
| ConvNeXt-B | 384x384 | 86.8 | 89M | 47.0G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth)
| ConvNeXt-L | 224x224 | 86.6 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth)  | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth)
| ConvNeXt-L | 384x384 | 87.5 | 198M | 101.0G |    -         | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth)
| ConvNeXt-XL | 224x224 | 87.0 | 350M | 60.9G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth) | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth)
| ConvNeXt-XL | 384x384 | 87.8 | 350M | 179.0G |  -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth)


### ImageNet-1K trained models (isotropic)
| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-S | 224x224 | 78.7 | 22M | 4.3G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 82.0 | 87M | 16.9G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth) |
| ConvNeXt-L | 224x224 | 82.6 | 306M | 59.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth) |


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation
We give an example evaluation command for a ImageNet-22K pre-trained, then ImageNet-1K fine-tuned ConvNeXt-B:

Single-GPU
```
python main.py --model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```

This should give 
```
* Acc@1 85.820 Acc@5 97.868 loss 0.563
```

- For evaluating other model variants, change `--model`, `--resume`, `--input_size` accordingly. You can get the url to pre-trained models from the tables above. 
- Setting model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in timm behaves the same during evaluation; but it is required in training. See [TRAINING.md](TRAINING.md) or our paper for the values used for different models.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```
-->