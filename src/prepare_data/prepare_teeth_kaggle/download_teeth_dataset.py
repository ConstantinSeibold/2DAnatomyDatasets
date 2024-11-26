import os
import requests
from zipfile import ZipFile

# Define the URL and the target path
url = "https://www.kaggle.com/api/v1/datasets/download/humansintheloop/teeth-segmentation-on-dental-x-ray-images"
download_path = os.path.expanduser("./archive.zip")
extract_path = os.path.expanduser(os.getenv("TEETH_ROOT"))


def download_and_extract(url, download_path, extract_path):
    try:
        # Download the file
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP request errors
        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete.")

        # Extract the zip file
        print("Extracting files...")
        with ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Files extracted to {extract_path}")

    finally:
        # Remove the ZIP file
        if os.path.exists(download_path):
            os.remove(download_path)
            print("ZIP file deleted.")


# Run the function
download_and_extract(url, download_path, extract_path)
