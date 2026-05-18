export MONTGOMERY_ROOT_FOLDER="./datasets/montgomery/"
python src/prepare_data/prepare_montgomery/download_montgomery.py
python src/prepare_data/prepare_montgomery/prepare_montgomery_labels.py
python src/prepare_data/prepare_montgomery/prepare_montgomery_splits.py
