export LESAV_ROOT_FOLDER="./datasets/lesav/"
python src/prepare_data/prepare_lesav/download_lesav.py
python src/prepare_data/prepare_lesav/prepare_lesav_labels.py
python src/prepare_data/prepare_lesav/prepare_lesav_splits.py
