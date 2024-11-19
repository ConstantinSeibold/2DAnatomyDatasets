import os
import zipfile
import gdown
import shutil

# Define the Google Drive file URL or ID
file_id = '1ZlZoSStvE9VCRq3bJiGhQH931EF0h3hh'  # Replace with your file ID
destination = 'ravir.zip'  # Destination path for the downloaded ZIP file

# Download the file from Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)

# Unzip the file
with zipfile.ZipFile(destination, 'r') as zip_ref:
    zip_ref.extractall("ravir_tmp")  # Destination folder for unzipped contents

# import pdb; pdb.set_trace()
shutil.copytree(os.path.join("ravir_tmp", "RAVIR Dataset"), os.getenv("RAVIR_ROOT_FOLDER"))
shutil.rmtree("ravir_tmp")
# Delete the downloaded ZIP file
if os.path.exists(destination):
    os.remove(destination)
    print("ZIP file deleted.")

print("Download, extraction, and deletion completed successfully.")

