import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

if __name__=="__main__":

    # Sourced from http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/segmentation02.zip
    jsrt_splits_root = os.getenv("JSRT_SEGMENTATION_ORIG_PATH")
    jsrt_img_root = os.getenv("JSRT_IMG_PATH")
    jsrt_mask_root = os.getenv("JSRT_MASK_PATH")
    jsrt_json_path = os.getenv("JSRT_JSON_PATH")

    train = "segmentation/label_train"
    test = "segmentation/label_test"

    train_val_ids = sorted(os.listdir(os.path.join(jsrt_splits_root, train)))

    train_ids = train_val_ids[:-len(train_val_ids)//10]
    val_ids = train_val_ids[-len(train_val_ids)//10:]
    test_ids = os.listdir(os.path.join(jsrt_splits_root, test))

    train_ids = [int(i.replace("_label.png","").replace("case","")) for i in train_ids]
    val_ids = [int(i.replace("_label.png","").replace("case","")) for i in val_ids]
    test_ids = [int(i.replace("_label.png","").replace("case","")) for i in test_ids]

    images = sorted(os.listdir(jsrt_img_root))
    images = {i+1:img for i,img in enumerate(images)}

    name = "JSRT"
    label_dict = {
        0: "Heart",
        1:	"Left Clavicle",
        2:	"Left Lung",
        3: "Right Clavicle",
        4: "Right Lung",
    }

    train = [
        {
            "target": os.path.join("masks",images[i].replace(".png",".npy")),
            "image": os.path.join("images",images[i])
        }
        for i in train_ids
    ]
    val = [
        {
            "target": os.path.join("masks",images[i].replace(".png",".npy")),
            "image": os.path.join("images",images[i])
        }
        for i in val_ids
    ]
    test = [
        {
            "target": os.path.join("masks",images[i].replace(".png",".npy")),
            "image": os.path.join("images",images[i])
        }
        for i in test_ids
    ]

    out = {
        "name": name,
        "label_dict": label_dict,
        "train": train,
        "val": val,
        "test": test,
    }

    with open(jsrt_json_path, "w") as f:
        json.dump(out, f)