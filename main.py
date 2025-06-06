import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.caml import CAML

def main(mode=None):
    
    config = load_config(mode)

    # CUDA e.g. 0,1,2,3
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # INIT GPU
    if torch.cuda.is_available() and len(config.GPU) > 1:
        torch.cuda.set_device(config.local_rank)
        config.DEVICE = torch.device("cuda", config.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    if torch.cuda.is_available() and len(config.GPU) == 1:
        config.DEVICE = torch.device("cuda")
        print('\nGPU IS AVAILABLE\n')
        torch.backends.cudnn.benchmark = True

    cv2.setNumThreads(0)

    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    model = CAML(config)
    
    model.load()
    
    if config.MODE == 1:
        print("Start Training...\n")
        model.train() 
    if config.MODE == 2:
        print("Start Testing...\n")
        model.test()


def load_config(mode=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--valiter', type=int, default=0, help='the iterator of val')
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # load config file
    config = Config(config_path)
    config.local_rank = args.local_rank

    # train mode
    if mode == 1:
        config.MODE = 1
        config.VAL_ITERS = args.valiter

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.VAL_ITERS = args.valiter

    return config


if __name__ == "__main__":
    main()