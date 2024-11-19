import os
import zipfile
import gdown

DATASET_ROOT_PATH = os.getenv("JSRT_ROOT_PATH")

# Define the Google Drive file URL or ID
file_id = '1L5QOTuqQ9OjTwP0hol_mjUDMWQPcLHSA'  # Replace with your file ID
destination = 'jsrt.zip'  # Destination path for the downloaded ZIP file

# Download the file from Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)

# Unzip the file
with zipfile.ZipFile(destination, 'r') as zip_ref:
    zip_ref.extractall("jsrt_tmp")  # Destination folder for unzipped contents

# import pdb; pdb.set_trace()
shutil.copytree(os.path.join("jsrt_tmp", "jsrt"),DATASET_ROOT_PATH)
shutil.rmtree("jsrt_tmp")

# Delete the downloaded ZIP file
if os.path.exists(destination):
    os.remove(destination)
    print("ZIP file deleted.")

print("Download, extraction, and deletion completed successfully.")
