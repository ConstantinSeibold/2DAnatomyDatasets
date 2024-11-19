export BS80K_ROOT='./datasets/bs80k'
python src/prepare_data/prepare_bs80k/download_bs80k.py
python src/prepare_data/prepare_bs80k/download_bs80k_anatomy.py
python src/prepare_data/prepare_bs80k/aggregate_images.py
