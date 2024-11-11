import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class JSRT_binary_Dataset(Dataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        splits = json.load(open(splits_json))
        
        self.label_dict = {
            0: "Heart",
            1:	"Left Clavicle",
            2:	"Left Lung",
            3: "Right Clavicle",
            4: "Right Lung",
        }
        
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

        if self.transform:
            mask = mask.transpose([1,2,0])
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.transpose(2,0,1)
            
        return image, mask