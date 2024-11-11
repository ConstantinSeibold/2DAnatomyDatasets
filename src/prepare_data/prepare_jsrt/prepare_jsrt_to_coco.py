# https://github.com/chrise96/image-to-coco-json-converter/tree/master

from PIL import Image
import numpy as np
from skimage import measure
import os
import json
from pycocotools import mask
from copy import deepcopy
from tqdm import tqdm

def binary_mask_to_rle(binary_mask: np.array) -> dict:
    """
    Convert binary mask to COCO RLE format.

    Args:
        binary_mask (np.array): Binary mask array.

    Returns:
        dict: COCO RLE encoded mask.
    """
    mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    mask_encoded['counts'] = mask_encoded['counts'].decode('utf-8')
    return mask_encoded
                      
def rle_to_binary_mask(rle_mask: dict) -> np.array:
    """
    Convert COCO RLE encoded mask to binary mask.

    Args:
        rle_mask (dict): COCO RLE encoded mask.

    Returns:
        np.array: Binary mask array.
    """
    rle_mask_copy = deepcopy(rle_mask)
    rle_mask_copy['counts'] = rle_mask_copy['counts'].encode('utf-8')
    binary_mask = mask.decode(rle_mask_copy)
    return binary_mask
  
def mask_to_annotation(mask: np.array, base_ann_id: int = 1, img_id: int = 1) -> list:
    """
    Convert mask array to COCO annotation format.

    Args:
        mask (np.array): Mask array.
        base_ann_id (int, optional): Base annotation ID. Defaults to 1.
        img_id (int, optional): Image ID. Defaults to 1.

    Returns:
        list: List of COCO annotations.
    """
    annotations = []
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            continue
        binary_mask = mask[i]
        mask_encoded = binary_mask_to_rle(binary_mask)
        annotation = {
            'id': base_ann_id + i,
            'image_id': img_id,
            'category_id': i + 1,
            'segmentation': mask_encoded,
            'area': int(np.sum(binary_mask)),
            'bbox': toBox(mask_encoded).tolist(),
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations

def toBox(binary_mask: np.array) -> list:
    """
    Convert binary mask to bounding box coordinates.

    Args:
        binary_mask (np.array): Binary mask array.

    Returns:
        list: Bounding box coordinates.
    """
    return mask.toBbox(binary_mask)

def create_category_annotation(category_dict):
    """
    Create category annotations in COCO JSON format.

    Args:
        category_dict (dict): Dictionary containing category names and IDs.

    Returns:
        list: List of category annotations.
    """
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def get_coco_json_format():
    """
    Get the standard COCO JSON format.

    Returns:
        dict: COCO JSON format skeleton.
    """
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [],
        "annotations": []
    }

    return coco_format

def get_image_info(path, id):
    image_info = {}
    width, height = Image.open(path).size
    image_info["id"] = id
    image_info["width"] = width
    image_info["height"] = height
    image_info["file_name"] = path.split("/")[-1]
    return image_info

def process_images(images, label_dict):
    train_json = get_coco_json_format()
    train_json["categories"] = [{"id": int(key)+1, "supercategory": "", "name":label_dict[key]} for key in label_dict.keys()]
    annotation_id = 1
    for i, image in enumerate(tqdm(images)):
        train_json["images"] += [get_image_info(image["image"], i)]
        mask = np.load(image["target"])
        annotations = mask_to_annotation(mask, annotation_id, i)
        train_json["annotations"] += annotations
        annotation_id += len(annotations)
    return train_json

def save_json(coco_dict, path, split):
    with open(os.path.join(path, split+".json"), "w") as f:
        json.dump(coco_dict, f)

if __name__=="__main__":
    jsrt_json_path = os.getenv("JSRT_JSON_PATH")
    jsrt_coco_path = os.getenv("JSRT_COCO_PATH")
        
    jsons = json.load(open(jsrt_json_path))
    
    label_dict = jsons["label_dict"]
    
    os.makedirs(jsrt_coco_path, exist_ok=True)
    
    train_json = process_images(jsons["train"], label_dict)
    save_json(train_json,jsrt_coco_path,"jsrt_train")
    
    val_json = process_images(jsons["val"], label_dict)
    save_json(val_json,jsrt_coco_path,"jsrt_val")
    
    test_json = process_images(jsons["test"], label_dict)
    save_json(test_json,jsrt_coco_path,"jsrt_test")
    
    
    
    