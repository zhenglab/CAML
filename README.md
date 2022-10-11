# CAML

This repository provides the official PyTorch implementation of our paper "Context-Aware Mutual Learning for Blind Image Inpainting and Beyond".

## Prerequisites

- Linux
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

