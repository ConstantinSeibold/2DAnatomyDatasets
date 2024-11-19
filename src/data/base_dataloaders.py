import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from PIL import Image
import numpy as np

class BaseMultiLabelDataset(Dataset):
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
        mask = np.load(mask_path)
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
        mask = np.array(Image.open(image_path).convert("RGB"))

        if self.transform is not None:
            mask = mask[0]
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]
            mask = transformed["mask"]
            
        return image, mask
        
class BaseDetectionDataset(CocoDetection):
    def __init__(self, root_dir, annFile, transform=None, target_transform= None, transforms = None):
        super(BaseDetectionDataset, self).__init__(root_dir, annFile,  transform, target_transform, transforms)
        self.label_dict = {self.coco.cats[key]["id"]:self.coco.cats[key]["name"] for key in self.coco.cats.keys()}
