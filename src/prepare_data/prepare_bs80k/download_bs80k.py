import os
import zipfile
import gdown

# Define the Google Drive file URL or ID
file_id = "1aoIwyxXSEyXDP9An-b2fsUNsZpAjxAVK"  # Replace with your file ID
destination = "bs80k.zip"  # Destination path for the downloaded ZIP file

# Download the file from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# Unzip the file
with zipfile.ZipFile(destination, "r") as zip_ref:
    zip_ref.extractall(
        os.getenv("BS80K_ROOT")
    )  # Destination folder for unzipped contents

# Delete the downloaded ZIP file
if os.path.exists(destination):
    os.remove(destination)
    print("ZIP file deleted.")

print("Download, extraction, and deletion completed successfully.")
