export HRF_ROOT_FOLDER="./datasets/hrf/"
python src/prepare_data/prepare_hrf/download_hrf.py
python src/prepare_data/prepare_hrf/prepare_hrf_labels.py
python src/prepare_data/prepare_hrf/prepare_hrf_splits.py
