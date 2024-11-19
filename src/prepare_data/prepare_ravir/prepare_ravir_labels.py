import numpy as np
import cv2
import os
from tqdm import tqdm

def remap_labels(image_path, label_mapping):
    """
    Remap the pixel values of an image based on a label mapping dictionary.

    Parameters:
        image_path (str): Path to the image to be remapped.
        label_mapping (dict): Dictionary where keys are original labels and values are new labels.

    Returns:
        None. The modified image is saved back to the original path.
    """
    # Load the image (assume single-channel grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Remap the labels using NumPy for fast pixel-wise operation
    remapped_image = np.copy(image)
    for original_value, new_value in label_mapping.items():
        remapped_image[image == original_value] = new_value

    # Save the remapped image back to the same path
    cv2.imwrite(image_path, remapped_image)

# Example Usage
def process_ravir_dataset(mask_folder, label_mapping):
    """
    Process all mask images in a folder and remap their labels.

    Parameters:
        mask_folder (str): Path to the folder containing mask images.
        label_mapping (dict): Dictionary mapping original labels to new labels.

    Returns:
        None. The modified images are saved back to their original paths.
    """
    # Get all image files in the folder
    mask_files = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for mask_path in tqdm(mask_files):
        remap_labels(mask_path, label_mapping)

# Define label mapping and folder path



if __name__=="__main__":
    root_folder = os.getenv("RAVIR_ROOT_FOLDER")
    folder = os.path.join(root_folder, "train", "training_masks")
    
    label_mapping = {
        0: 0,  # Background remains the same
        128: 1,  # Map label 128 to 1
        255: 2,  # Map label 255 to 2
    }
        
    process_ravir_dataset(folder, label_mapping)