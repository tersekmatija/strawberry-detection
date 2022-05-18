import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxsim
import subprocess
import blobconverter

from utils.draw import draw_overlay
from utils.config import load_config
from utils.boxutils import non_max_suppression
import utils.augmentations as A
from datasets.strawberrydi import StrawDIDataset
from models.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

cfg = load_config(args.config)

model = Model(cfg.num_classes, cfg.anchors, cfg.strides, cfg.reduction, export=True)


#if cfg.demo_weights is None:
#    raise RuntimeError("Demo run not set!")
#state_dict = torch.load(cfg.demo_weights, map_location="cpu")


#model.load_state_dict(state_dict)
model.cpu()
model.eval()

print(model.det_head.detect)

inp = torch.rand(1,3,*cfg.img_shape)
output_names = ["segmentation", *['output'+str(i+1)+'_yolov5' for i in range(model.det_head.detect.nl)]]
input_names = ["input"]
torch.onnx.export(model, inp, "model.onnx", opset_version=11, input_names = input_names, output_names = output_names)

model_onnx = onnx.load("model.onnx")  # load onnx model
onnx_model, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'
onnx.save(onnx_model, "model.onnx")


cmd = f"mo --input_model model.onnx " \
        f"--output_dir output " \
        f"--model_name model " \
        '--data_type FP16 ' \
        '--scale 255 ' #\
        #'--reverse_input_channel ' \
        
        #f'--output "{output_list}"'

subprocess.check_output(cmd, shell=True)

blob_path = blobconverter.from_openvino(
            xml="./output/model.xml",#as_posix(),
            bin="./output/model.bin",#as_posix(),
            data_type="FP16",
            shaves=6,
            version="2021.4",
            use_cache=False,
            output_dir="./output/"
        )

os.rename(blob_path, "./output/model.blob")