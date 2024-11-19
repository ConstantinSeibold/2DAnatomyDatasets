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
│   └── prepare_teeth_kaggle/                   # Scripts for processing the Teeth dataset
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

---

## 🚀 **Getting Started**

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/2D-Anatomy-Segmentation-Datasets.git
cd 2D-Anatomy-Segmentation-Datasets
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
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
This project is licensed under the [MIT License](LICENSE).  

---

## 💬 **Contact**
For questions or feedback, feel free to reach out:  
📧 Email: constantinseibold[at]gmail.com  
🔗 [GitHub Issues](https://github.com/ConstantinSeibold/2DAnatomyDatasets/issues)  

---

Enjoy building robust segmentation models with these datasets! 😊  