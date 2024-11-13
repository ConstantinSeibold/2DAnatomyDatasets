export PAXRAY_JSON_PATH="./datasets/paxray"
export PAXRAY_COCO_PATH="./datasets/paxray/jsons"
python src/prepare_data/prepare_paxray/download_paxray.py
python src/prepare_data/prepare_paxray/prepare_paxray_to_coco.py