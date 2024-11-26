import os
import json
import random
from sklearn.model_selection import train_test_split

# Duke dataset label dictionary for OCT layers
label_dictionary = {
    0: "Background",
    1: "RPE",
    2: "ILM",
    3: "NFL",
    4: "GCL",
    5: "IPL",
    6: "INL",
    7: "OPL",
    8: "ONL",
    9: "ELM",
    10: "IS",
    11: "OS",
    12: "RPE",
    13: "BM",
    14: "Choroid",
    15: "Sclera",
}


def create_duke_dataset_splits(
    root_folder, dataset_name, validation_split=0.2, output_file="duke_splits.json"
):
    # Define paths to images and labels folders in the Duke dataset
    images_path = os.path.join(root_folder, "images")
    labels_path = os.path.join(root_folder, "labels")

    # Get list of image files and corresponding label files
    image_files = sorted(
        [
            os.path.join(images_path, f)
            for f in os.listdir(images_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    label_files = sorted(
        [
            os.path.join(labels_path, f)
            for f in os.listdir(labels_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Ensure that the number of images and labels match
    if len(image_files) != len(label_files):
        raise ValueError("Number of images and labels do not match!")

    train_images_split, test_images, train_labels_split, test_labels = train_test_split(
        image_files, label_files, test_size=validation_split, random_state=42
    )

    # Split into train and validation sets
    train_images_split, val_images_split, train_labels_split, val_labels_split = (
        train_test_split(
            train_images_split,
            train_labels_split,
            test_size=validation_split,
            random_state=42,
        )
    )

    # Create splits dictionary
    splits = {
        "name": dataset_name,
        "label_dict": label_dictionary,
        "train": [
            {
                "image": img.replace(root_folder, ""),
                "target": lbl.replace(root_folder, ""),
            }
            for img, lbl in zip(train_images_split, train_labels_split)
        ],
        "val": [
            {
                "image": img.replace(root_folder, ""),
                "target": lbl.replace(root_folder, ""),
            }
            for img, lbl in zip(val_images_split, val_labels_split)
        ],
        "test": [
            {
                "image": img.replace(root_folder, ""),
                "target": lbl.replace(root_folder, ""),
            }
            for img, lbl in zip(test_images, test_labels)
        ],
    }

    # Save the splits to a JSON file
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Duke dataset splits saved to {output_file}")


if __name__ == "__main__":

    dataset_folder = os.getenv("DUKE_ROOT_PATH")

    # Example usage
    create_duke_dataset_splits(
        root_folder=dataset_folder,
        dataset_name="Duke_OCT",
        validation_split=0.2,
        output_file=os.path.join(dataset_folder, "duke_splits.json"),
    )
