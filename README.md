https://user-images.githubusercontent.com/56075061/166344196-dbc344b8-e050-443d-9aa7-9520fe90c8ef.png

# :strawberry: Strawberry detection :heavy_plus_sign: segmentation 

Model is based on YOLOP, which consist of a mutual encoder/decoder like structure, with detection and segmentation head.

## Training and data preparation

To train the model, download the dataset from [StrawDI](https://strawdi.github.io/) website, put it into `data/` folder, and modify the config.

In config, configure model parameters, anchors, strides, and other necessary settings.

To compute anchors call:
```
python anchors.py -cfg path/to/config.yaml
```
Each anchor consist of 2 numbers, representing width and height. When pasting anchors to config, you have to manually group them. In Yolo architecture, they are typically made of groups of 3 anchors, denoting 3 anchors per detection branch in detection head. The model was pretrained with 2 detection heads only, and consequently `3 * 2 = 6` anchors. 

To train, use:
```
python train.py -cfg path/to/config.yaml
```

Start a tensorboard with `--logdir runs` to track the training progress and inspect the validation images.

## Inference
To test the model on the test set define `demo_run: name_of_run` in config and call:
```
python demo.py -cfg path/to/config.yaml
```

## Export for OAK devices
To export and generate a blob for OAK device call:
```
python export.py -cfg path/to/config.yaml
```

## Config settings
Config allows you to configure a number of settings:
```
# Train settings
dataset: "strawdi"              # name of the dataset, currently only supported
batch_size: 2                   # batch size used for training
epochs: 20                      # number of epochs
optimizer: "adam"               # optimizier, supported 'adam' and 'sgd'
learning_rate: 0.0001           # initial learning rate
dataset_dir: "./data/StrawDI_Db1"   # path to dataset
runs_dir: "runs/"               # path where experiments will be saved
warmup: 'linear'                # warmup for the CosineAnnealing scheduler
warmup_iters: 50                # number of warmup iterations
momentum: 0.937                 # momentum used by Adam (first beta) or SGD
weight_decay: 0.00005           # L2 regularization
pretrained:                     # path to a pretrained model, leave empty if none
num_workers: 6                  # number of workers used for data loading

# Train parts - configure which parts should be trained and which are frozen
backbone: True
decoder: True                   
head_seg: False
head_det: True

# Parameters for weighting different losses
w_box: 0.05
w_cls: 0.3
w_obj: 1
w_iou: 0.1
w_seg: 0.1

# Validation settings
val_plot_num: 4                 # Number of images per dataset to plot in tensorboard
thr_iou: 0.6                    # IoU threshold for object detector
thr_conf: 0.3                   # Confidence threshold for object detector

# Model settings
reduction: 4                    # Use default_num_channels/reduction channels. Higher reduction, less channels the models uses, the faster (but less accurate) it is
num_classes: 1                  # Number of classes in the dataset

# Define anchors and number of heads in object detector
# Anchors can be computed using anchors.py script
# Higher stride, deeper features are used (8, 16, 32) supported.
# 3 head version
#anchors: [[8,15,  14,23,  20,35],  [26,51,  31,73,  42,60],  [40,95,  49,118,  61,151]]
#strides: [8,16,32]
# 2 head version
anchors: [[13,22,  21,35,  31,60], [41,94,  55,129,  73,174]]
strides: [16,32]
# 1 head version
# anchors: [[10,17,  19,33,  28,56,  37,87,  51,125]]
# strides: [32]

# Augmentations used for training
img_shape: [480, 640]               # Image size
blur_p: 0.4                         # Blur probability
blur_ks: 10                         # Blur kernel size
flip_p: 0.5                         # Horizontal flip probability
rotate_p: 15                        # Rotation angle
min_scale: 0.9                      # Minimum scaling factor

# Demo
demo_run: "exp63"                   # Name of the experiment for inference
```

## Custom dataset suppport
Define a custom data loader in `dataset/` directory and set it up in `get_loader` function in `dataset/loaders`. Take current data loader as an example. A custom collate method is used, as expected output for a batch is `img, [sem, boxes]`, where: 
* `img.shape = [N, 3, H, W]`,
* `sem.shape = [N, 2, H, W]` ,

where `N = batch_size`, `H = image_height`, and `W = image_width`. Boxes have a shape of `[boxes_in_batch, 6]`, where each row consist of `img_id_in_batch, class, x_center, y_center, w, h` using normalized coordinates.

## TODO

- [ ] Evaluation script (mAP)
- [ ] Edge demo

## Credits

This repo contains the code for strawberry detection and segmentations. Parts of code are based on repositories of [YOLOP](https://github.com/hustvl/YOLOP) and [YoloV5](https://github.com/ultralytics/yolov5/).
