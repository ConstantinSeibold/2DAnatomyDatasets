import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from PIL import Image
import numpy as np
from typing import Any, Callable, Optional, Tuple, List

class BaseMultiLabelDataset(Dataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        splits = json.load(open(splits_json))
        
        self.label_dict = {int(key):splits["label_dict"][key] for key in splits["label_dict"]}
        
        self.splits = splits[split]
        
    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        current_case = self.splits[idx]
        
        image_path = os.path.join(self.root_dir, current_case["image"])
        mask_path = os.path.join(self.root_dir, current_case["target"])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.load(mask_path)>0

        mask = mask[list(self.label_dict.keys())].astype(int)

        if self.transform is not None:
            mask = mask.transpose([1,2,0])
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.permute(2,0,1)
            
        return image, mask
    
class BaseMultiClassDataset(Dataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        splits = json.load(open(splits_json))
        
        self.label_dict = splits["label_dict"]
        
        self.splits = splits[split]
        
    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        current_case = self.splits[idx]
        
        image_path = os.path.join(self.root_dir, current_case["image"])
        mask_path = os.path.join(self.root_dir, current_case["target"])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("F"))

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]
            mask = transformed["mask"]
            
        return image, mask
        
class BaseDetectionDataset(CocoDetection):
    def __init__(self, root_dir, annFile, transform=None, target_transform= None, transforms = None):
        super(BaseDetectionDataset, self).__init__(root_dir, annFile,  transform, target_transform, transforms)
        self.label_dict = {self.coco.cats[key]["id"]:self.coco.cats[key]["name"] for key in self.coco.cats.keys()}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        masks = []
        bboxes = []
        labels = []
        for t in target:
            masks.append(self.coco.annToMask(t))
            bboxes += [t["bbox"]]
            labels += [t["category_id"]]
        
        masks = np.stack(masks, axis=-1) if masks else np.zeros((img.shape[0], img.shape[1], 0), dtype=np.uint8)

        if self.transforms is not None:
            image = image.convert("RGB")
            image = np.array(image)#[:,:,0]
            masks = masks.transpose([2,0,1])
            
            augmented = self.transforms(image=image, bboxes=bboxes, masks=masks, labels=labels)
            image = augmented['image']
            bboxes = augmented['bboxes']
            masks = augmented['masks']
            labels = augmented["labels"]
            target = {
                'image': torch.tensor(image),
                'boxes': torch.tensor(bboxes, dtype=torch.float32),
                'masks': torch.stack(masks, 0),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'coco': target
            }
        else:
            target = {
                'image': torch.tensor(np.array(image)),
                'boxes': torch.tensor(bboxes, dtype=torch.float32),
                'masks': torch.tensor(masks),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'coco': target
            }

        
        return image, target