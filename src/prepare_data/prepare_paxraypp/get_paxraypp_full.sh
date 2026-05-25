export PAXRAYPP_ROOT="./datasets/paxraypp"
python src/prepare_data/prepare_paxraypp/download_paxraypp.py
python src/prepare_data/prepare_paxraypp/unpack_files.py
python src/prepare_data/prepare_paxraypp/prepare_paxraypp_splits.py