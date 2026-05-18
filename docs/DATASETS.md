# Datasets

Auto-generated from `src/anatomy_datasets/registry.py`. Do not edit by hand —
update the registry and re-run `python scripts/gen_dataset_docs.py`.

For each dataset the table records imaging modality, segmentation task type,
license, upstream source, and primary citation. Where the upstream license is
ambiguous, the entry is marked `unknown` rather than guessed; consult the
linked source before using the data in a commercial setting.

Full BibTeX entries for every dataset live in
[`CITATIONS.bib`](CITATIONS.bib) and are also expanded per-row below.

| Name | Modality | Task | License | Source | Citation |
|------|----------|------|---------|--------|----------|
| **BS80k** | scintigraphy | multilabel_segmentation | unknown | [source](https://pubmed.ncbi.nlm.nih.gov/36334360/) | Huang et al., BS-80K: The first large open-access dataset of bone scan images, Computers in Biology and Medicine 2022. [[paper]](https://doi.org/10.1016/j.compbiomed.2022.106221) |
| **CHASE_DB1** | fundus | multiclass_segmentation | [custom-academic](https://blogs.kingston.ac.uk/retinal/chasedb1/) | [source](https://blogs.kingston.ac.uk/retinal/chasedb1/) | Fraz et al., An ensemble classification-based approach applied to retinal blood vessel segmentation, IEEE TBME 2012. [[paper]](https://doi.org/10.1109/TBME.2012.2205687) |
| **DRIVE** | fundus | multiclass_segmentation | [custom-academic](https://drive.grand-challenge.org/) | [source](https://drive.grand-challenge.org/) | Staal et al., Ridge-based vessel segmentation in color images of the retina, IEEE TMI 2004. [[paper]](https://doi.org/10.1109/TMI.2004.825627) |
| **DUKE** | oct | multiclass_segmentation | [custom-academic](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm) | [source](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm) | Chiu et al., Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema, Biomedical Optics Express 2015. [[paper]](https://doi.org/10.1364/BOE.6.001172) |
| **JSRT** | chest_xray | multilabel_segmentation | [custom-academic](http://db.jsrt.or.jp/eng.php) | [source](http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/segmentation02.zip) | Shiraishi et al., Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule, AJR 2000. [[paper]](https://doi.org/10.2214/ajr.174.1.1740071) |
| **JumpBroadcast** | rgb_video | multiclass_segmentation | [custom-academic](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/) | [source](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/) | Ludwig et al., 2023. All Keypoints You Need: Detecting Arbitrary Keypoints on the Body of Triple, High, and Long Jump Athletes. CVPRW. [[paper]](https://arxiv.org/abs/2304.02939) |
| **MedakaHeart** | microscopy | multiclass_segmentation | [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) | [source](https://osf.io/uyk79/) | Schutera et al., Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels, PLOS ONE 2022. [[paper]](https://doi.org/10.1371/journal.pone.0263656) |
| **Montgomery** | chest_xray | multiclass_segmentation | [custom-academic](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html) | [source](https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip) | Jaeger et al., Two public chest X-ray datasets for computer-aided screening of pulmonary diseases, Quantitative Imaging in Medicine and Surgery 2014. [[paper]](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20) |
| **PAXRay** | chest_xray | multilabel_segmentation | [CC-BY-NC-SA-4.0](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation/blob/main/LICENSE.txt) | [source](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation) | Seibold et al., Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding, BMVC 2022. [[paper]](https://arxiv.org/abs/2210.03416) |
| **PAXRay++** | chest_xray | multilabel_segmentation | [CC-BY-NC-SA-4.0](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation/blob/main/LICENSE.txt) | [source](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation) | Seibold et al., Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling, 2023. [[paper]](https://arxiv.org/abs/2306.03934) |
| **RAVIR** | ir_retina | multiclass_segmentation | [custom-academic](https://ravir.grand-challenge.org/) | [source](https://ravir.grand-challenge.org/) | Hatamizadeh et al., RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging, JBHI 2022. [[paper]](https://doi.org/10.1109/JBHI.2022.3163352) |
| **STARE** | fundus | multiclass_segmentation | unknown | [source](https://cecas.clemson.edu/~ahoover/stare/) | Hoover et al., Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response, IEEE TMI 2000. [[paper]](https://doi.org/10.1109/42.845178) |
| **Teeth** | panoramic_xray | detection | [custom-academic](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) | [source](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) | Humans in the Loop, Teeth Segmentation on Dental X-Ray Images (Kaggle). |

## Notes per dataset

- **BS80k**: 82,544 bone scan images / 3,247 patients from West China Hospital. Hosted on Google Drive by the authors; no explicit dataset license — treat as research-use-only and contact authors for any redistribution.
- **CHASE_DB1**: First 20 of 28 cases used for train (with val carved out via seed=42); remaining 8 are test.
- **DRIVE**: Test partition ships images without vessel annotations; test entries in the splits JSON have target=null.
- **DUKE**: Random split with seed=42 (~64/16/20).
- **JSRT**: Uses the upstream official train/test split; train is sliced 90/10 into train/val by index (no random seed needed).
- **JumpBroadcast**: Upstream test annotations gated; we re-stratify athletes 70/15/15 with seed=42 so the public release has a usable test split.
- **MedakaHeart**: Raw data + labels redistributed via OSF project uyk79 under GPL-3.0. 565 annotated train frames; two unseen-specimen test sets pooled into the test split.
- **Montgomery**: 138 PA chest radiographs (80 normal, 58 TB) with bilateral lung masks shipped as ManualMask/leftMask + rightMask. No official split; random 80/10/10 with seed=42. Labels: 0=bg, 1=left_lung, 2=right_lung.
- **PAXRay**: 166-class multilabel chest X-ray anatomy; splits JSON ships with the upstream download.
- **PAXRay++**: Extended PAXRay variant.
- **RAVIR**: Test partition has no public ground truth; downstream eval must skip or submit to the challenge.
- **STARE**: STARE ships 397 raw images but only 20 are manually annotated. Split is random with seed=42 (10/4/6).
- **Teeth**: Polygon annotations converted to COCO format. Random split with seed=42 (70/10/20).

## BibTeX

Aggregate file: [`CITATIONS.bib`](CITATIONS.bib). Per-dataset entries:

<details><summary><b>BS80k</b></summary>

```bibtex
@article{huang2022bs80k,
    title   = {{BS-80K}: The first large open-access dataset of bone scan images},
    author  = {Huang, Zongmo and Pu, Xiaorong and Tang, Gongshun and Ping, Ming and Jiang, Guo and Wang, Mengjie and Wei, Xiaoyu and Ren, Yazhou},
    journal = {Computers in Biology and Medicine},
    volume  = {151},
    pages   = {106221},
    year    = {2022},
    doi     = {10.1016/j.compbiomed.2022.106221},
}
```

</details>

<details><summary><b>CHASE_DB1</b></summary>

```bibtex
@article{fraz2012chasedb1,
    title   = {An ensemble classification-based approach applied to retinal blood vessel segmentation},
    author  = {Fraz, Muhammad Moazam and Remagnino, Paolo and Hoppe, Andreas and Uyyanonvara, Bunyarit and Rudnicka, Alicja R. and Owen, Christopher G. and Barman, Sarah A.},
    journal = {IEEE Transactions on Biomedical Engineering},
    volume  = {59},
    number  = {9},
    pages   = {2538--2548},
    year    = {2012},
    doi     = {10.1109/TBME.2012.2205687},
}
```

</details>

<details><summary><b>DRIVE</b></summary>

```bibtex
@article{staal2004drive,
    title   = {Ridge-based vessel segmentation in color images of the retina},
    author  = {Staal, Joes and Abr{\`a}moff, Michael D. and Niemeijer, Meindert and Viergever, Max A. and van Ginneken, Bram},
    journal = {IEEE Transactions on Medical Imaging},
    volume  = {23},
    number  = {4},
    pages   = {501--509},
    year    = {2004},
    doi     = {10.1109/TMI.2004.825627},
}
```

</details>

<details><summary><b>DUKE</b></summary>

```bibtex
@article{chiu2015duke,
    title   = {Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema},
    author  = {Chiu, Stephanie J. and Allingham, Michael J. and Mettu, Priyatham S. and Cousins, Scott W. and Izatt, Joseph A. and Farsiu, Sina},
    journal = {Biomedical Optics Express},
    volume  = {6},
    number  = {4},
    pages   = {1172--1194},
    year    = {2015},
    doi     = {10.1364/BOE.6.001172},
}
```

</details>

<details><summary><b>JSRT</b></summary>

```bibtex
@article{shiraishi2000jsrt,
    title   = {Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule: Receiver Operating Characteristic Analysis of Radiologists' Detection of Pulmonary Nodules},
    author  = {Shiraishi, Junji and Katsuragawa, Shigehiko and Ikezoe, Junpei and Matsumoto, Tsuneo and Kobayashi, Takeshi and Komatsu, Ken-ichi and Matsui, Mitate and Fujita, Hiroshi and Kodera, Yoshie and Doi, Kunio},
    journal = {American Journal of Roentgenology},
    volume  = {174},
    number  = {1},
    pages   = {71--74},
    year    = {2000},
    doi     = {10.2214/ajr.174.1.1740071},
}
```

</details>

<details><summary><b>JumpBroadcast</b></summary>

```bibtex
@inproceedings{ludwig2023all_kps,
    title     = {All Keypoints You Need: Detecting Arbitrary Keypoints on the Body of Triple, High, and Long Jump Athletes},
    author    = {Ludwig, Katja and Lorenz, Julian and Sch{\"o}n, Robin and Lienhart, Rainer},
    booktitle = {Proceedings of the 2023 IEEE/CVF International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    month     = {June},
    year      = {2023},
}
```

</details>

<details><summary><b>MedakaHeart</b></summary>

```bibtex
@article{schutera2022frugal,
    title   = {Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels},
    author  = {Schutera, Mark and Rettenberger, Luca and Pylatiuk, Christian and Reischl, Markus},
    journal = {PLOS ONE},
    volume  = {17},
    number  = {2},
    pages   = {e0263656},
    year    = {2022},
    doi     = {10.1371/journal.pone.0263656},
}
```

</details>

<details><summary><b>Montgomery</b></summary>

```bibtex
@article{jaeger2014montgomery,
    title   = {Two public chest {X-ray} datasets for computer-aided screening of pulmonary diseases},
    author  = {Jaeger, Stefan and Candemir, Sema and Antani, Sameer and W{\'a}ng, Yi-Xi{\'a}ng J. and Lu, Pu-Xuan and Thoma, George},
    journal = {Quantitative Imaging in Medicine and Surgery},
    volume  = {4},
    number  = {6},
    pages   = {475--477},
    year    = {2014},
    doi     = {10.3978/j.issn.2223-4292.2014.11.20},
}
```

</details>

<details><summary><b>PAXRay</b></summary>

```bibtex
@inproceedings{seibold2022paxray,
    title     = {Detailed Annotations of Chest {X-Rays} via {CT} Projection for Report Understanding},
    author    = {Seibold, Constantin and Rei{\ss}, Simon and Sarfraz, M. Saquib and Stiefelhagen, Rainer and Kleesiek, Jens},
    booktitle = {Proceedings of the 33rd British Machine Vision Conference (BMVC)},
    year      = {2022},
    eprint    = {2210.03416},
    archivePrefix = {arXiv},
}
```

</details>

<details><summary><b>PAXRay++</b></summary>

```bibtex
@article{seibold2023paxraypp,
    title         = {Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling},
    author        = {Seibold, Constantin and Rei{\ss}, Simon and Sarfraz, Saquib and Fink, Matthias A. and Mayer, Victoria and Sellner, Jan and Kim, Moon Sung and Maier-Hein, Klaus H. and Kleesiek, Jens and Stiefelhagen, Rainer},
    journal       = {arXiv preprint arXiv:2306.03934},
    year          = {2023},
    eprint        = {2306.03934},
    archivePrefix = {arXiv},
}
```

</details>

<details><summary><b>RAVIR</b></summary>

```bibtex
@article{hatamizadeh2022ravir,
    title   = {{RAVIR}: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging},
    author  = {Hatamizadeh, Ali and Hosseini, Hamid and Patel, Niraj and Choi, Jinseo and Pole, Cameron C. and Hoeferlin, Cory M. and Schwartz, Steven D. and Terzopoulos, Demetri},
    journal = {IEEE Journal of Biomedical and Health Informatics},
    volume  = {26},
    number  = {7},
    pages   = {3272--3283},
    year    = {2022},
    doi     = {10.1109/JBHI.2022.3163352},
}
```

</details>

<details><summary><b>STARE</b></summary>

```bibtex
@article{hoover2000stare,
    title   = {Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response},
    author  = {Hoover, A. D. and Kouznetsova, Valentina and Goldbaum, Michael},
    journal = {IEEE Transactions on Medical Imaging},
    volume  = {19},
    number  = {3},
    pages   = {203--210},
    year    = {2000},
    doi     = {10.1109/42.845178},
}
```

</details>

<details><summary><b>Teeth</b></summary>

```bibtex
@misc{hitl2023teeth,
    title        = {Teeth Segmentation on Dental {X-Ray} Images},
    author       = {{Humans in the Loop}},
    year         = {2023},
    howpublished = {Kaggle dataset},
    url          = {https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images},
}
```

</details>

