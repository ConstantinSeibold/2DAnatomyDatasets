import os
import json
import random
from sklearn.model_selection import train_test_split

def create_dataset_splits(root_folder, dataset_name, label_dictionary, validation_split=0.2, output_file="splits.json"):
    # Define paths
    train_images_path = os.path.join(root_folder, "train", "training_images")
    train_masks_path = os.path.join(root_folder, "train", "training_masks")
    test_images_path = os.path.join(root_folder, "test")

    # Get list of training images and masks
    train_images = sorted([os.path.join(train_images_path, f) for f in os.listdir(train_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    train_masks = sorted([os.path.join(train_masks_path, f) for f in os.listdir(train_masks_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Ensure the number of images matches the number of masks
    if len(train_images) != len(train_masks):
        raise ValueError("Number of training images and masks do not match!")

    # Split into train and validation sets
    train_images_split, val_images_split, train_masks_split, val_masks_split = train_test_split(
        train_images, train_masks, test_size=validation_split, random_state=42
    )

    # Get test images and optionally masks if available
    test_images = sorted([os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    test_masks = sorted([os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    test_data = []
    for img in test_images:
        mask = os.path.join(test_images_path, os.path.basename(img)) if os.path.basename(img) in test_masks else None
        test_entry = {"image": img}
        if mask:
            test_entry["target"] = mask
        test_data.append(test_entry)

    # Create splits dictionary
    splits = {
        "name": dataset_name,
        "label_dictionary": label_dictionary,
        "train": [{"image": img, "target": mask} for img, mask in zip(train_images_split, train_masks_split)],
        "val": [{"image": img, "target": mask} for img, mask in zip(val_images_split, val_masks_split)],
        "test": test_data
    }

    # Save splits to JSON file
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Dataset splits saved to {output_file}")

if __name__=="__main__":
    root_folder = os.getenv("RAVIR_ROOT_FOLDER")
    create_dataset_splits(
        root_folder=root_folder,
        dataset_name="RAVIR",
        label_dictionary={0: "background", 1: "arteries",2: "veins"},
        validation_split=0.2,
        output_file= os.path.join(root_folder,"ravir_splits.json")
    )
