from collections import defaultdict
from .base_dataloaders import BaseMultiLabelDataset, BaseDetectionDataset


class PAXRay4_binary_Dataset(BaseMultiLabelDataset):
    def __init__(self, root_dir, splits_json, split, transform=None):
        super(PAXRay4_binary_Dataset, self).__init__(
            root_dir, splits_json, split, transform
        )

        self.label_dict = {
            0: "lungs_overall",
            10: "mediastinum_overall",
            24: "bones",
            163: "diaphragm",
        }


class PAXRay4_detection_Dataset(BaseDetectionDataset):
    def __init__(
        self, root_dir, annFile, transform=None, target_transform=None, transforms=None
    ):
        super(BaseDetectionDataset, self).__init__(
            root_dir, annFile, transform, target_transform, transforms
        )

        from pycocotools.coco import COCO

        # +1 because coco ids start at 1
        self.label_dict = {
            0 + 1: "lungs_overall",
            10 + 1: "mediastinum_overall",
            24 + 1: "bones",
            163 + 1: "diaphragm",
        }

        self.coco.dataset["annotations"] = [
            ann
            for ann in self.coco.dataset["annotations"]
            if ann["category_id"] in self.label_dict.keys()
        ]

        self.coco.createIndex()


PAXRay166_binary_Dataset = BaseMultiLabelDataset
PAXRay166_detection_Dataset = BaseDetectionDataset
