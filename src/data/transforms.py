import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform_det(split="train"):
    if split=="train":
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.ElasticTransform(p=0.1),
            A.GridDistortion(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask', "bboxes":"bboxes", }, is_check_shapes=False, bbox_params=A.BboxParams(format='coco', label_fields=['labels',]))
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def get_transform(split="train"):
    if split=="train":
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.ElasticTransform(p=0.1),
            A.GridDistortion(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask', }, is_check_shapes=False)
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
