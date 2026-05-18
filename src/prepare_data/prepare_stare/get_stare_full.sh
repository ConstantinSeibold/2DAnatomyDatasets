export STARE_ROOT_FOLDER="./datasets/stare/"
python src/prepare_data/prepare_stare/download_stare.py
python src/prepare_data/prepare_stare/prepare_stare_labels.py
python src/prepare_data/prepare_stare/prepare_stare_splits.py
