import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import scipy.io
import glob
import shutil
from PIL import Image


def get_valid_idx(manual_layer: np.ndarray) -> list[int]:
    """
    Identify valid index layers where data is present.

    Args:
        manual_layer (np.ndarray): Array of manual layers data.

    Returns:
        list[int]: List of indices with non-zero layer data.
    """
    return [
        i for i in range(manual_layer.shape[2]) if np.sum(manual_layer[:, :, i]) != 0
    ]


def get_valid_img_seg_reimpl(scan_obj: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Process scan data to extract and validate image and segmentation labels.

    Args:
        scan_obj (dict): Data structure from .mat file containing scan data.

    Returns:
        tuple[np.ndarray, np.ndarray]: Image data and segmentation label arrays.
    """
    fluid_class = 9
    manual_layer = np.array(scan_obj["manualLayers1"], dtype=np.uint16)
    manual_fluid = np.array(scan_obj["manualFluid1"], dtype=np.uint16)
    img = np.array(scan_obj["images"], dtype=np.uint8)
    valid_idx = get_valid_idx(manual_layer)

    manual_fluid = manual_fluid[:, :, valid_idx]
    manual_layer = manual_layer[:, :, valid_idx]

    seg = np.zeros_like(manual_fluid, dtype=np.uint8)

    for bsc in range(seg.shape[2]):
        for asc in range(seg.shape[1]):
            class_idx = manual_layer[:, asc, bsc]
            for i in range(1, len(class_idx)):
                if class_idx[i] < class_idx[i - 1]:
                    class_idx[i] = class_idx[i - 1]
            for label, (idx_prev, idx_cur) in enumerate(
                zip([0, *class_idx], [*class_idx, seg.shape[0]])
            ):
                seg[idx_prev:idx_cur, asc, bsc] = label

    seg[manual_fluid > 0] = fluid_class
    (a_scan_used,) = np.where(np.sum(manual_layer, axis=(0, 2)) != 0)
    seg = seg[:, a_scan_used[0] : a_scan_used[-1] + 1]
    img = img[:, a_scan_used[0] : a_scan_used[-1] + 1]
    img = img[:, :, valid_idx]

    return img, seg


def create_pipeline(paths: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Pipeline to process and compile images and labels from multiple files.

    Args:
        paths (list[str]): List of file paths to .mat files.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: Compiled image and label arrays.
    """
    images, labels = [], []
    for file_path in tqdm(paths):
        data = scipy.io.loadmat(file_path)
        img_data, label_data = get_valid_img_seg_reimpl(data)
        for idx in range(img_data.shape[-1]):
            images.append(img_data[:, :, idx])
            labels.append(label_data[:, :, idx])
    return images, labels


def load_and_save_dataset():
    """Load dataset from .mat files, process, and save as images and labels."""
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    file_paths = glob.glob(f"{dataset_folder}/2015_BOE_Chiu/*.mat")
    images, labels = create_pipeline(file_paths)

    for i, (image, label) in enumerate(zip(images, labels)):
        Image.fromarray(image.astype(np.uint8)).save(
            os.path.join(output_images_folder, f"image_{i}.png")
        )
        Image.fromarray(label.astype(np.uint8)).save(
            os.path.join(output_labels_folder, f"label_{i}.png")
        )

    shutil.rmtree(f"{dataset_folder}/2015_BOE_Chiu/")

    print("Image and label saving complete.")
    print("Dataset processing complete. Images and labels saved.")


if __name__ == "__main__":

    # Define output folders
    dataset_folder = os.getenv("DUKE_ROOT_PATH")
    output_images_folder = os.path.join(dataset_folder, "images")
    output_labels_folder = os.path.join(dataset_folder, "labels")
    load_and_save_dataset()
