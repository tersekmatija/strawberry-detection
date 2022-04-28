import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from YoloV5
class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)