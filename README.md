# CAML

This repository provides the official PyTorch implementation of our paper "Context-Aware Mutual Learning for Blind Image Inpainting and Beyond".

## More Results

- Qualitative comparison results of unseen contaminated patterns for blind image inpainting.

![](./imgs/unseen_contaminated_patterns.pdf)

## Prerequisites

![Alt text](https://vscode-remote%252Bssh-002dremote-002b8.vscode-resource.vscode-cdn.net/data/zhr/acmm_blind/checkpoints/place2_sota_aot_new_graffity/results0_tcsvt_graft/output/Places365_val_00003607.jpg?version%253D1691329185259)- Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## Getting Started


### Installation

- Clone this repo:
```bash
git clone https://github.com/zhenglab/CAML.git
cd CAML
```

- Install [PyTorch](http://pytorch.org) and 1.4 and other dependencies (e.g., torchvision).
  - For Conda users, you can create a new Conda environment using `conda create --name <env> --file requirements.txt`.

### Training

```
python train.py --path=$configpath$

For example: python train.py --path=./checkpoints/
```

### Testing

```
python test.py --path=$configpath$ 

For example: python test.py --path=./checkpoints/
```

