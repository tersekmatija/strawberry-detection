import argparse
from models.model import Model
from models.layers import C3
from utils.loss import *
from utils.config import *
import torch

from fvcore.nn import FlopCountAnalysis, flop_count_table

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

cfg = load_config(args.config)


inp = torch.rand(1, 3, *cfg.img_shape)
#inp = torch.rand(1, 3, 640, 640)
print(inp.shape)
print(cfg)
model = Model(cfg.num_classes, cfg.anchors, cfg.strides, cfg.reduction)
#model.eval()
model.train()

flops = FlopCountAnalysis(model, inp)
print(flop_count_table(flops, max_depth=1))

model.eval()
model(inp)
