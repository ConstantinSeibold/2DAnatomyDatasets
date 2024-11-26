import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import shutil


def convert_to_coco(ann_dir, img_dir, output_file):
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": "Converted Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-11-18",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Category mapping and ID generator
    category_id_map = {}
    annotation_id = 0  # Unique ID for annotations

    files = sorted(os.listdir(ann_dir))
    # Process each annotation file
    for ann_file in tqdm(files):
        if ann_file.endswith(".json"):
            # Load annotation JSON
            ann_path = os.path.join(ann_dir, ann_file)
            with open(ann_path, "r") as f:
                ann_data = json.load(f)

            # Extract image filename and size
            filename = ann_file.replace(".json", "")
            image_path = os.path.join(img_dir, filename)
            if not os.path.exists(image_path):
                print(f"Image file {filename} not found, skipping...")
                continue

            image_id = len(coco_format["images"]) + 1
            image_info = {
                "id": image_id,
                "file_name": filename,
                "height": ann_data["size"]["height"],
                "width": ann_data["size"]["width"],
            }
            coco_format["images"].append(image_info)

            # Process objects in the annotation
            for obj in ann_data.get("objects", []):
                class_title = obj["classTitle"]
                if class_title not in category_id_map:
                    category_id = int(class_title)
                    category_id_map[class_title] = category_id
                    coco_format["categories"].append(
                        {
                            "id": category_id,
                            "name": class_title,
                            "supercategory": "none",
                        }
                    )
                else:
                    category_id = category_id_map[class_title]

                # Convert polygon points
                points = obj["points"]["exterior"]
                segmentation = [
                    coord for point in points for coord in point
                ]  # Flatten list of points

                # Calculate bounding box (xmin, ymin, width, height)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                xmin, ymin = min(x_coords), min(y_coords)
                width, height = max(x_coords) - xmin, max(y_coords) - ymin

                # Add annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": width * height,
                    "bbox": [xmin, ymin, width, height],
                    "iscrowd": 0,
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

    # Save COCO JSON to output file
    with open(output_file, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO dataset saved to {output_file}")
    return coco_format


def create_splits(coco_data, output_dir, splits=(0.7, 0.1, 0.2)):
    random.seed(42)  # For reproducibility
    images = coco_data["images"]
    random.shuffle(images)

    # Calculate split indices
    total_images = len(images)
    train_count = int(splits[0] * total_images)
    val_count = int(splits[1] * total_images)
    train_images = images[:train_count]
    val_images = images[train_count : train_count + val_count]
    test_images = images[train_count + val_count :]

    # Helper to filter annotations by image IDs
    def filter_annotations(image_ids):
        return [ann for ann in coco_data["annotations"] if ann["image_id"] in image_ids]

    # Create split datasets
    splits_data = {
        "train": {
            "images": train_images,
            "annotations": filter_annotations({img["id"] for img in train_images}),
            "categories": coco_data["categories"],
        },
        "val": {
            "images": val_images,
            "annotations": filter_annotations({img["id"] for img in val_images}),
            "categories": coco_data["categories"],
        },
        "test": {
            "images": test_images,
            "annotations": filter_annotations({img["id"] for img in test_images}),
            "categories": coco_data["categories"],
        },
    }

    # Save splits
    for split_name, split_data in splits_data.items():
        split_path = os.path.join(output_dir, f"{split_name}.json")
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=4)
        print(f"Saved {split_name} split to {split_path}")

    return splits_data


def create_class_agnostic(coco_data, output_dir, splits_data):
    for split_name, split_data in splits_data.items():
        # Replace all categories with a single class
        agnostic_data = {
            "images": split_data["images"],
            "annotations": [
                {**ann, "category_id": 1} for ann in split_data["annotations"]
            ],
            "categories": [{"id": 1, "name": "classagnostic", "supercategory": "none"}],
        }
        # Save class-agnostic file
        agnostic_path = os.path.join(output_dir, f"{split_name}_classagnostic.json")
        with open(agnostic_path, "w") as f:
            json.dump(agnostic_data, f, indent=4)
        print(f"Saved class-agnostic {split_name} to {agnostic_path}")


# Example usage
if __name__ == "__main__":
    teeth_root = os.getenv("TEETH_ROOT")
    ann_folder = os.path.join(teeth_root, "Teeth Segmentation JSON/d2/ann")
    img_folder = os.path.join(teeth_root, "Teeth Segmentation JSON/d2/img")
    output_coco_file = os.path.join(teeth_root, "teeth_coco.json")

    # Convert to COCO format
    coco_data = convert_to_coco(ann_folder, img_folder, output_coco_file)

    # Create train/val/test splits
    splits = create_splits(coco_data, teeth_root, splits=(0.7, 0.1, 0.2))

    # Create class-agnostic files
    create_class_agnostic(coco_data, teeth_root, splits)

    shutil.move(
        os.path.join(teeth_root, "Teeth Segmentation JSON/d2/img"),
        os.path.join(teeth_root, "img"),
    )

    shutil.rmtree(os.path.join(teeth_root, "Teeth Segmentation JSON"))
    shutil.rmtree(os.path.join(teeth_root, "Teeth Segmentation PNG"))
