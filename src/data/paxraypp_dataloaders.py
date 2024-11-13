import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from . import BaseDataset

class PAXRayPP_binary_Dataset(BaseDataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        super(PAXRayPP_binary_Dataset, self).__init__(root_dir, splits_json, split, transform)
        
class PAXRayPP_Instance_Dataset(BaseDataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        super(PAXRayPP_Instance_Dataset, self).__init__(root_dir, splits_json, split, transform)