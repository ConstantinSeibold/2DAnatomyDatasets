import os
import urllib.request
import zipfile
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATASET_URL = "http://www.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip"
DATASET_ROOT_PATH = os.getenv("DUKE_ROOT_PATH")
DATASET_FOLDER = "2015_BOE_Chiu2"
ZIP_PATH = f"{DATASET_FOLDER}.zip"


def download_and_extract_dataset():
    """Downloads and extracts the dataset if it does not already exist.

    The dataset is downloaded as a zip file and extracted into the directory
    specified by DATASET_ROOT_PATH. If the dataset zip file or the extracted
    folder already exists, it skips the relevant steps.
    """
    if DATASET_ROOT_PATH is None:
        logger.error("Environment variable 'DUKE_ROOT_PATH' is not set.")
        return

    # Download the dataset if zip file doesn't exist
    if not os.path.exists(ZIP_PATH):
        logger.info("Downloading dataset from %s...", DATASET_URL)
        try:
            urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        except Exception as e:
            logger.error("Failed to download dataset: %s", e)
            return
        logger.info("Download complete.")
    else:
        logger.info("Dataset zip file already exists.")

    # Extract the dataset if folder doesn't exist
    dataset_extract_path = os.path.join(DATASET_ROOT_PATH, DATASET_FOLDER)
    if not os.path.exists(dataset_extract_path):
        logger.info("Extracting dataset...")
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(DATASET_ROOT_PATH)
            logger.info("Extraction complete.")
        except zipfile.BadZipFile:
            logger.error("Failed to extract dataset. The zip file may be corrupted.")
            return
        finally:
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
                logger.info("Zip file removed after extraction.")
    else:
        logger.info("Dataset folder already exists.")


if __name__ == "__main__":
    download_and_extract_dataset()
    logger.info("Download and extraction process complete.")
