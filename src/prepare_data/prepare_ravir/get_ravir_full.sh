export RAVIR_ROOT_FOLDER="./datasets/ravir/"
python src/prepare_data/prepare_ravir/download_ravir.py
python src/prepare_data/prepare_ravir/prepare_ravir_labels.py
python src/prepare_data/prepare_ravir/prepare_ravir_splits.py