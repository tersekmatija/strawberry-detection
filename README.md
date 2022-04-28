# Strawberry detection

## Training and data preparation

To train the model, download the dataset from [StrawDI](https://strawdi.github.io/) website, put it into `data/` folder, and modify the config.

To train, use:
```
python train.py -cfg path/to/config.yaml
```

Start a tensorboard with `--logdir runs` to track the training progress and inspect the validation images.

## TODO

- [] Evaluation script (mAP)
- [] Export script
- [] Edge demo

## Credits

This repo contains the code for strawberry detection and segmentations. Parts of code are based on repositories of [YOLOP](https://github.com/hustvl/YOLOP) and [YoloV5](https://github.com/ultralytics/yolov5/).
