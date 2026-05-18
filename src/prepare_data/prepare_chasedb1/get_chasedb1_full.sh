export CHASEDB1_ROOT_FOLDER="./datasets/chasedb1/"
python src/prepare_data/prepare_chasedb1/download_chasedb1.py
python src/prepare_data/prepare_chasedb1/prepare_chasedb1_labels.py
python src/prepare_data/prepare_chasedb1/prepare_chasedb1_splits.py
