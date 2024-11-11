# Anatomy in Chest X-Ray - Datasets

Welcome to the PAXRay and PAXRay++ datasets! These datasets provide a detailed foundation for segmentation of anatomical structures in X-ray images, utilizing projections of 3D CT scans to simulate high-quality 2D radiographs with fine-grained anatomical labels. This README explains the datasets' content, structure, and provides resources such as data loaders and basic training scripts for segmentation models.

---

## Table of Contents
- [Overview](#overview)
    - [JSRT](#JSRT)
    - [PAXRay](#paxray)
    - [PAXRay++](#paxray++)
- [Access](#access)
- [Usage](#usage)
- [Citation](#citation)

---

## Overview

### JSRT Database

<details>
  <summary> Expand for details of JSRT</summary>
The Standard Digital Image Database of chest radiographs with and without a lung nodule was developed as part of the research activities of the Academic Committee of the Japanese Society of Radiological Technology (JSRT) in 1998 (*1), supported by the Japan Radiological Society (JRS) and with the cooperation of healthcare facilities in Japan and the United States for providing case materials. 

- **247 images** in both frontal and lateral views.
- **5 anatomical classes** available [here](documentation/labelset_jsrt.md).

</details>


<details>
  <summary> Expand for Data Licence </summary>
</details>


### PAXRay


<details>
  <summary> Expand for details of PAXRay-4 </summary>
  
  PAXRay is based on the RibFrac CT dataset and provides **880 X-ray-like images** by projecting CT data to a 2D plane, imitating traditional X-ray imagery. The dataset is equipped with **multi-label segmentation masks**:
- **880 images** in both frontal and lateral views.
    - ``train``: 
    - ``validation``:
    - ``test``: 
- **4 anatomical classes** that include lungs, bones, mediastinum, diaphragm. The  categories are available [here](documentation/labelset_paxray.md).
</details>

<details>
  <summary> Expand for details of PAXRay-166 </summary>
  
  \
  PAXRay is based on the RibFrac CT dataset and provides **880 X-ray-like images** by projecting CT data to a 2D plane, imitating traditional X-ray imagery. The dataset is equipped with **multi-label segmentation masks**:
- **880 images** in both frontal and lateral views.
    - ``train``: 
    - ``validation``:
    - ``test``: 
- **166 anatomical classes** that include both fine-grained (92 individual anatomical structures) and super-class categories available [here](documentation/labelset_paxray.md).

</details>


<details>
  <summary> Expand for Data Information </summary>
  
#### Data Access

- [gDrive](https://drive.google.com/drive/folders/1rzlsZ0bfByRMBoywOPWZW08GNgIwCU9P?usp=sharing)

#### Dataset Structure

`File Naming Convention`

Each image and its corresponding mask file share the same base name for easy pairing. For example:

| Path | Description |
| :--- | :----------
| [paxray_dataset](https://drive.google.com/drive/folders/1rzlsZ0bfByRMBoywOPWZW08GNgIwCU9P?usp=sharing) | Main folder.
| &boxvr;&nbsp; images | Contains the proposed loss formulation for both Tensorflow and Pytorch
| &boxv;&nbsp; &ensp;&ensp; &boxvr;&nbsp; RibFrac1frontal.png | projected CT image of shape 512x512
| &boxv;&nbsp; &ensp;&ensp; &boxvr;&nbsp; ... | 
| &boxvr;&nbsp; labels | Contains scripts for recreating the FCC dataset and data preparations for training/testing
| &boxv;&nbsp; &ensp;&ensp; &boxvr;&nbsp; RibFrac1frontal.npy | binary segmentation masks in numpy format of shape 166x512x512
| &boxv;&nbsp; &ensp;&ensp; &boxvr;&nbsp; ... | 
| paxray_half.json | associated supplimentary notes (85.6 MB)
| paxray_quarter.json | associated supplimentary notes (85.6 MB)
| paxray_half.json | associated supplimentary notes (85.6 MB)

</details>

<details>
  <summary> Expand for Data Licence </summary>
</details>



### PAXRay++

<details>
  <summary> Expand for details of PAXRay++ </summary>
PAXRay++ expands on PAXRay, introduced by Seibold et al., and is constructed using pseudo-labeled thoracic CTs. This dataset is ideal for fine-grained segmentation of anatomy within chest X-rays.
- **7,377 images** in both frontal and lateral views.
- **157 anatomical classes** and over **2 million annotated instances** across the dataset available [here](documentation/labelset_paxray++.md).
</details>

---




---

## Access

---

## Usage

### Data Loading

To facilitate model training and evaluation, we provide PyTorch data loaders for PAXRay and PAXRay++. These data loaders will handle the structure of both datasets, allowing easy access to images and masks.

#### Prerequisites
Ensure you have `torch` and `torchvision` installed:
```bash
pip install torch torchvision
```

## Citation

If you use PAXRay or PAXRay++ in your research, please cite the relevant papers as follows:

```bibtex

@article{paxray,
  title = {PAXRay: A Projected Dataset for the Segmentation of Anatomical Structures in X-Ray Data},
  author = {Authors},
  journal = {Journal Name},
  year = {Year},
  pages = {xx--xx}
}

@article{paxrayplusplus,
  title = {Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling},
  author = {Seibold et al.},
  journal = {Journal Name},
  year = {Year},
  pages = {xx--xx}
}
```