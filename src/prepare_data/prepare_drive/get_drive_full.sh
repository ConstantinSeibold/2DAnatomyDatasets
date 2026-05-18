export DRIVE_ROOT_FOLDER="./datasets/drive/"
python src/prepare_data/prepare_drive/download_drive.py
python src/prepare_data/prepare_drive/prepare_drive_labels.py
python src/prepare_data/prepare_drive/prepare_drive_splits.py
