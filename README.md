# 2D Anatomy Segmentation Datasets Repository  

Welcome to the **2D Anatomy Segmentation Datasets** repository! This project provides a collection of scripts, tools, and datasets to streamline training and evaluation for anatomical segmentation tasks in medical imaging. Whether you're a researcher or practitioner in the medical imaging field, this repository is designed to collect 2D anatomical datasets and make it easy to download, process, and visualize data from various imaging modalities.

---

## 📦 **Datasets Included**

This repository supports the following publicly available datasets:

| Dataset         | Modality         | #Images               | Description                                                                                      | Link                             |
|------------------|------------------|--------------------------------|--------------------------------------------------------------------------------------------------|----------------------------------|
| **BS80k**       | Scintigraphy    |       6,494         | Large-scale dataset for anatomy segmentation in nuclear imaging.                                   | [Link](https://pubmed.ncbi.nlm.nih.gov/36334360/)                       |
| **JSRT**        | Chest X-Ray   |        247         | Dataset of chest radiographs for lung segmentation.                                              | [Link](db.jsrt.or.jp/eng.php)                       |
| **PAX-Ray**     | Chest X-Ray    |       852         | Dataset for fine-grained anatomy  segmentation.                                                  | [Link](https://constantinseibold.github.io/paxray.html)                       |
| **PAX-Ray++**   | Chest X-Ray    |      14,754         | Enhanced version of PAX-Ray with additional annotations.                                         | [Link](https://constantinseibold.github.io/paxray.html)                       |
| **DUKE**        | Optical Coherence Tomography  | 110 | OCT dataset for retinal layer segmentation.                                                     | [Link](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm)                       |
| **RAVIR**       | Infrared Reflectance Imaging   | 23 |Dataset for retinal vessel segmentation from infrared images.                                    | [Link](https://ravir.grand-challenge.org/data/)                       |
| **Teeth**       | Panoramic X-Ray     |     598       | Dataset of dental X-rays for teeth segmentation.                                                 | [Link](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/discussion?sort=published)                       |
| **DRIVE**       | Color Fundus     |     40        | Retinal vessel segmentation dataset (20 train / 20 test).                                        | [Link](https://www.isi.uu.nl/Research/Databases/DRIVE/)                       |
| **STARE**       | Color Fundus     |     20        | Retinal vessel segmentation dataset with two manual annotators.                                  | [Link](https://cecas.clemson.edu/~ahoover/stare/)                       |
| **CHASE_DB1**   | Color Fundus     |     28        | Retinal vessel segmentation dataset from school-age children.                                    | [Link](https://researchdata.kingston.ac.uk/96/)                       |
| **Jump-Broadcast** | RGB Video (broadcast) | 1,809 | Cropped body-part segmentation masks for triple/high/long jump athletes (15 classes). | [Link](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/) |
| **MedakaHeart** | Brightfield Microscopy (Medaka hatchling) | 805 | Ventral-view heart segmentation (bulbus / atrium / heart) — 565 train + two held-out specimens (N0030: 75, R0004: 165). GPL-3.0. | [Link](https://osf.io/uyk79/) |
| **Montgomery**  | Chest X-Ray      |    138        | NLM Montgomery County chest X-rays with bilateral lung masks (left / right lung as separate classes). Random 80/10/10 split. | [Link](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html) |

> For **license, citation, source URL, BibTeX, and split-seed information per dataset**, see [`docs/DATASETS.md`](docs/DATASETS.md) and the aggregate [`docs/CITATIONS.bib`](docs/CITATIONS.bib). Both are auto-generated from `src/anatomy_datasets/registry.py` — regenerate with `python scripts/gen_dataset_docs.py` whenever the registry changes. The same metadata (including BibTeX) is also embedded in each dataset's splits JSON so it can be queried programmatically.

---

## 📁 **Repository Structure**  
Here’s an overview of the repository's structure:

```plaintext
src/
│
├── data/                        # PyTorch dataloaders for all datasets
│
├── prepare_data/
│   ├── prepare_bs80k/                   # Scripts for processing the anatomical segmentations of BS80k dataset
│   ├── prepare_jsrt/                    # Scripts for processing the JSRT dataset
│   ├── prepare_paxray/                 # Scripts for processing the PAX-Ray dataset
│   ├── prepare_paxraypp/               # Scripts for processing the PAX-Ray++ dataset
│   ├── prepare_duke/                    # Scripts for processing the DUKE dataset
│   ├── prepare_ravir/                   # Scripts for processing the RAVIR dataset
│   ├── prepare_teeth_kaggle/                   # Scripts for processing the Teeth dataset
│   ├── prepare_drive/                   # Scripts for processing the DRIVE retinal vessel dataset
│   ├── prepare_stare/                   # Scripts for processing the STARE retinal vessel dataset
│   ├── prepare_chasedb1/                # Scripts for processing the CHASE_DB1 retinal vessel dataset
│   ├── prepare_jump_broadcast/          # Scripts for processing the Jump-Broadcast dataset
│   ├── prepare_medaka_heart/            # Scripts for processing the Medaka hatchling heart dataset
│   └── prepare_montgomery/              # Scripts for processing the NLM Montgomery lung dataset
│
├── visualization/
│   └── visualize.py             # Utility scripts for visualizing dataset samples
│
├── Notebooks/
│   └── Download_datasets.ipynb    # shows how to setup all the datasets
│   └── Dataloader_example.ipynb   # Examples for to setup the dataloaders for each dataset and visualize annotations

```


---

## ⚙️ **Features**

### 🛠 **Data Preparation**
Each dataset includes standardized processing scripts to ensure consistency across datasets. These scripts:  
- Download the raw data.  
- Normalize and preprocess the images.  
- Prepare data splits (train/val/test).  
- Create PyTorch-friendly formats.

### 📊 **Data Visualization**
Use the scripts in `src/visualization` to visualize the datasets, ground-truth labels, and predictions for easy inspection of data quality.

### 🚀 **Training Pipeline Integration**
Prepared datasets can be plugged directly into any PyTorch-based segmentation model for training and evaluation.

### 🔌 **Use with mmseg / mmdet / SMP / HuggingFace**
After `pip install -e .[<extra>]`, each dataset is consumable through the major segmentation ecosystems. Per-framework one-pagers:

- [`docs/QUICKSTART_SMP.md`](docs/QUICKSTART_SMP.md) — segmentation_models_pytorch (no exporter, no adapter).
- [`docs/QUICKSTART_MMSEG.md`](docs/QUICKSTART_MMSEG.md) — runtime adapter + per-channel exporter for multilabel.
- [`docs/QUICKSTART_MMDET.md`](docs/QUICKSTART_MMDET.md) — monolithic COCO for small datasets, sharded COCO for PAXRay-scale.
- [`docs/QUICKSTART_HF.md`](docs/QUICKSTART_HF.md) — HuggingFace `datasets.DatasetDict` + `transformers.AutoImageProcessor`.

See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) for which datasets work with which framework, and [`docs/SCHEMA.md`](docs/SCHEMA.md) for the splits-JSON + sharded-COCO format specs. Copy-paste training configs live under [`configs/`](configs/).

---

## 🚀 **Getting Started**

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/2D-Anatomy-Segmentation-Datasets.git
cd 2D-Anatomy-Segmentation-Datasets
```

### 2️⃣ Install Dependencies  
Dependencies are declared in `pyproject.toml` (no `requirements.txt`):
```bash
# Core only (dataset prep + raw PyTorch dataloaders):
pip install -e .
# Or pick an ecosystem extra:
pip install -e '.[smp]'      # segmentation_models_pytorch
pip install -e '.[mmseg]'    # mmsegmentation adapter + exporter
pip install -e '.[mmdet]'    # mmdetection adapter + exporter
pip install -e '.[hf]'       # HuggingFace datasets
pip install -e '.[all]'      # everything
```

### 3️⃣ Prepare the Data  
Run the dataset preparation script for the dataset of interest. For example:  
```bash
sh src/prepare_data/prepare_bs80k/get_bs80k_full.sh

sh src/prepare_data/prepare_jsrt/get_jsrt_full.sh

...
```

### 4️⃣ Visualize the Data  
To visualize samples:  
```bash
python src/visualization/visualize.py --dataset BS80k
```

---

## 🤝 **Contributions**
Contributions are welcome! If you'd like to add support for more datasets, improve baseline performance, or enhance data preparation scripts, feel free to open a pull request.

---

## 📜 **License**
This project (scripts, dataloaders, docs) is licensed under the [MIT License](LICENSE). Each upstream dataset retains its own license — see [`docs/DATASETS.md`](docs/DATASETS.md) for the per-dataset license column and [`docs/CITATIONS.bib`](docs/CITATIONS.bib) for full BibTeX. The registry uses `license="unknown"` rather than guessing; check the source before commercial use.

## 📚 **Citing the datasets**
If you use any dataset prepared by this repo in published work, **cite the upstream paper**. Sources:

- Per-dataset license, source URL, paper DOI, and BibTeX: [`docs/DATASETS.md`](docs/DATASETS.md)
- Aggregate BibTeX file ready to drop into your reference manager: [`docs/CITATIONS.bib`](docs/CITATIONS.bib)
- Programmatic access from Python:
  ```python
  from anatomy_datasets import get_dataset_info
  info = get_dataset_info("DRIVE")
  print(info.citation, info.bibtex, info.license, info.source_url)
  ```
- Embedded inside every prepared splits JSON under the `citation` / `bibtex` keys, and printed by every `prepare_<name>_splits.py` run.

---

## 💬 **Contact**
For questions or feedback, feel free to reach out:  
📧 Email: constantinseibold[at]gmail.com  
🔗 [GitHub Issues](https://github.com/ConstantinSeibold/2DAnatomyDatasets/issues)  

---

Enjoy building robust segmentation models with these datasets! 😊  