export DUKE_ROOT_PATH="./datasets/duke"
python src/prepare_data/prepare_duke/download_duke.py
python src/prepare_data/prepare_duke/prepare_duke.py
python src/prepare_data/prepare_duke/prepare_duke_splits.py