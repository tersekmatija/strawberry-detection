# based on: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
from time import time
import multiprocessing as mp
import argparse

from utils.config import load_config
from datasets.loaders import get_loader
import utils.augmentations as A
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

np.testing.suppress_warnings()

cfg = load_config(args.config)

batch_size = cfg.batch_size

max_iters = 500

for num_workers in range(2, mp.cpu_count()+1, 2):  
    train_loader = get_loader(cfg.dataset, "train", cfg.dataset_dir, cfg.batch_size, transforms=None, num_workers=num_workers, pin_memory=True)
    start = time()
    for i, data in tqdm(enumerate(train_loader, 0)):
        if i > max_iters:
            break
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))