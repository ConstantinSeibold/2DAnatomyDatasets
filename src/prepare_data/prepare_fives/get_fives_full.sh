export FIVES_ROOT_FOLDER="./datasets/fives/"
python src/prepare_data/prepare_fives/download_fives.py
python src/prepare_data/prepare_fives/prepare_fives_labels.py
python src/prepare_data/prepare_fives/prepare_fives_splits.py
