import torch
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.boxutils import non_max_suppression


# expects non batched images
def draw_overlay(img_out, boxes_out, seg_out, show = True):
    
    img = (img_out * 255).type(torch.uint8)

    if seg_out is not None:
        seg = (seg_out > 0.5).bool() # 0.5 equal to argmax
        img = draw_segmentation_masks(img, seg, alpha=0.8, colors="blue")

    if boxes_out is not None:
        boxes_out = non_max_suppression(boxes_out, conf_thres=0.3, iou_thres=0.3, classes=None, agnostic=False)
        boxes_out = boxes_out[0]

        if len(boxes_out > 0):
            boxes = boxes_out[:, :4].int()
            #print(boxes)
            img = draw_bounding_boxes(img, boxes, colors="red")

    if show:
        img_show = img.numpy()
        img_show = img_show.astype(np.uint8).transpose(1,2,0)
        plt.imshow(img_show)
        plt.show()

    return img

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
