import zipfile
import tarfile
import os

if __name__ == "__main__": 
    PAXRAYPP_ROOT = os.getenv("PAXRAYPP_ROOT", "paxraypp")

    # File paths
    zip_file_path = 'labels.zip'
    tar_file_path = 'paxray_images_unfiltered.tar.gz'

    # Unpacking the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('labels_unpacked')  # Extract to a folder named "labels_unpacked"

    # Unpacking the tar.gz file
    with tarfile.open(tar_file_path, 'r:gz') as tar_ref:
        tar_ref.extractall('paxray_images_unfiltered')  # Extract to a folder named "paxray_images_unfiltered"

    # Deleting the original compressed files
    os.remove(zip_file_path)
    os.remove(tar_file_path)

    # Check the directories to confirm unpacking
    os.listdir('labels_unpacked'), os.listdir('paxray_images_unfiltered')
