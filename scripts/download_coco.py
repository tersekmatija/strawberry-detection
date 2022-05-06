import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import cv2
import random
import json

random.seed(0)
fo.config.default_ml_backend = "torch"

print("--- Downloading dataset ---")
downloaded_datasets = foz.list_downloaded_zoo_datasets()
if "coco-2017" in downloaded_datasets and "train" in downloaded_datasets["coco-2017"][1].downloaded_splits and "validation" in downloaded_datasets["coco-2017"][1].downloaded_splits:
    dataset_path = downloaded_datasets["coco-2017"][0]
else:
    # To download the COCO dataset for only the "person" and "car" classes
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "validation"],
        label_types=["detections", "segmentations"],
        classes=["person"],
    )
    dataset_path = downloaded_datasets["coco-2017"][0]

print("--- File check ---")
print(os.listdir(os.path.join(dataset_path, "train", "data"))[:5])

split_names = {}
for split in ["train", "validation"]:
    coco = COCO(os.path.join(dataset_path, "raw", f"instances_train2017.json")) if split == "train" else COCO(os.path.join(dataset_path, "raw", f"instances_val2017.json"))

    cat_ids = coco.getCatIds(["person"])
    img_ids = coco.getImgIds(catIds=cat_ids)
    img_infos = coco.loadImgs(img_ids)

    img_names = []

    for img_info in tqdm(img_infos):

        fn = img_info["file_name"]
        ann_ids = coco.getAnnIds(imgIds=img_info["id"], catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for i in range(len(anns)):
            mask[coco.annToMask(anns[i]) > 0] = i+1

        fn_mask = f"{fn.split('.')[0]}_label.png"

        cv2.imwrite(os.path.join(dataset_path, split, "data", fn_mask), mask)

        img_names.append(fn)

    if split == "train":
        random.shuffle(img_names)
        idx_break = int(len(img_names) * 0.9)
        split_names["train"] = img_names[:idx_break]
        split_names["val"] = img_names[idx_break:]
    else:
        split_names["test"] = img_names

print("--- Saving JSON splits ---")
with open(os.path.join(dataset_path, 'splits_person.json'), 'w', encoding='utf-8') as f:
    json.dump(split_names, f, ensure_ascii=False, indent=4)