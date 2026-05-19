from anatomy_datasets.base import BaseDetectionDataset, BaseMultiLabelDataset

LESAV_Vessel_Dataset = BaseMultiLabelDataset
LESAV_AV_Dataset = BaseMultiLabelDataset

# Detection variants: consumed with a COCO annFile produced by
# ``anatomy_datasets.exporters.to_coco(splits_json=..., root_dir=...)``,
# which RLE-encodes each non-empty multilabel channel as one COCO instance.
LESAV_Vessel_detection_Dataset = BaseDetectionDataset
LESAV_AV_detection_Dataset = BaseDetectionDataset
