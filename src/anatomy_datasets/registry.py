"""Static per-dataset metadata.

License / citation / source / modality are authored by hand here so they
can be queried programmatically (downstream consumers, doc generation)
without scraping the README.

When license information for an upstream dataset is uncertain, the entry
sets ``license="unknown"`` and records what is known under ``notes``. Do
**not** guess at licenses for datasets where the upstream source is
ambiguous - mislabeling a dataset's license is a worse failure mode than
flagging it as unknown.

Every entry must carry a ``bibtex`` block. This is the single source of
truth for the per-row BibTeX shown in the README dataset table and
embedded in every prepared splits JSON under ``bibtex``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    modality: str           # "chest_xray" | "fundus" | "oct" | "scintigraphy" | "panoramic_xray" | "ir_retina" | "microscopy" | "rgb_video"
    task: str               # "multiclass_segmentation" | "multilabel_segmentation" | "detection"
    license: str            # SPDX-ish identifier or "custom-academic" / "unknown"
    license_url: Optional[str]
    source_url: str
    citation: str           # short-form reference
    bibtex: str             # full BibTeX entry (single source of truth)
    paper_url: Optional[str] = None
    notes: str = ""


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "DRIVE": DatasetInfo(
        name="DRIVE",
        modality="fundus",
        task="multiclass_segmentation",
        license="custom-academic",
        license_url="https://drive.grand-challenge.org/",
        source_url="https://drive.grand-challenge.org/",
        citation="Staal et al., Ridge-based vessel segmentation in color images of the retina, IEEE TMI 2004.",
        paper_url="https://doi.org/10.1109/TMI.2004.825627",
        bibtex=r"""@article{staal2004drive,
    title   = {Ridge-based vessel segmentation in color images of the retina},
    author  = {Staal, Joes and Abr{\`a}moff, Michael D. and Niemeijer, Meindert and Viergever, Max A. and van Ginneken, Bram},
    journal = {IEEE Transactions on Medical Imaging},
    volume  = {23},
    number  = {4},
    pages   = {501--509},
    year    = {2004},
    doi     = {10.1109/TMI.2004.825627},
}""",
        notes="Test partition ships images without vessel annotations; test entries in the splits JSON have target=null.",
    ),
    "STARE": DatasetInfo(
        name="STARE",
        modality="fundus",
        task="multiclass_segmentation",
        license="unknown",
        license_url=None,
        source_url="https://cecas.clemson.edu/~ahoover/stare/",
        citation="Hoover et al., Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response, IEEE TMI 2000.",
        paper_url="https://doi.org/10.1109/42.845178",
        bibtex=r"""@article{hoover2000stare,
    title   = {Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response},
    author  = {Hoover, A. D. and Kouznetsova, Valentina and Goldbaum, Michael},
    journal = {IEEE Transactions on Medical Imaging},
    volume  = {19},
    number  = {3},
    pages   = {203--210},
    year    = {2000},
    doi     = {10.1109/42.845178},
}""",
        notes="STARE ships 397 raw images but only 20 are manually annotated. Split is random with seed=42 (10/4/6).",
    ),
    "CHASE_DB1": DatasetInfo(
        name="CHASE_DB1",
        modality="fundus",
        task="multiclass_segmentation",
        license="custom-academic",
        license_url="https://blogs.kingston.ac.uk/retinal/chasedb1/",
        source_url="https://blogs.kingston.ac.uk/retinal/chasedb1/",
        citation="Fraz et al., An ensemble classification-based approach applied to retinal blood vessel segmentation, IEEE TBME 2012.",
        paper_url="https://doi.org/10.1109/TBME.2012.2205687",
        bibtex=r"""@article{fraz2012chasedb1,
    title   = {An ensemble classification-based approach applied to retinal blood vessel segmentation},
    author  = {Fraz, Muhammad Moazam and Remagnino, Paolo and Hoppe, Andreas and Uyyanonvara, Bunyarit and Rudnicka, Alicja R. and Owen, Christopher G. and Barman, Sarah A.},
    journal = {IEEE Transactions on Biomedical Engineering},
    volume  = {59},
    number  = {9},
    pages   = {2538--2548},
    year    = {2012},
    doi     = {10.1109/TBME.2012.2205687},
}""",
        notes="First 20 of 28 cases used for train (with val carved out via seed=42); remaining 8 are test.",
    ),
    "RAVIR": DatasetInfo(
        name="RAVIR",
        modality="ir_retina",
        task="multiclass_segmentation",
        license="custom-academic",
        license_url="https://ravir.grand-challenge.org/",
        source_url="https://ravir.grand-challenge.org/",
        citation="Hatamizadeh et al., RAVIR: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging, JBHI 2022.",
        paper_url="https://doi.org/10.1109/JBHI.2022.3163352",
        bibtex=r"""@article{hatamizadeh2022ravir,
    title   = {{RAVIR}: A Dataset and Methodology for the Semantic Segmentation and Quantitative Analysis of Retinal Arteries and Veins in Infrared Reflectance Imaging},
    author  = {Hatamizadeh, Ali and Hosseini, Hamid and Patel, Niraj and Choi, Jinseo and Pole, Cameron C. and Hoeferlin, Cory M. and Schwartz, Steven D. and Terzopoulos, Demetri},
    journal = {IEEE Journal of Biomedical and Health Informatics},
    volume  = {26},
    number  = {7},
    pages   = {3272--3283},
    year    = {2022},
    doi     = {10.1109/JBHI.2022.3163352},
}""",
        notes="Test partition has no public ground truth; downstream eval must skip or submit to the challenge.",
    ),
    "DUKE": DatasetInfo(
        name="Duke_OCT",
        modality="oct",
        task="multiclass_segmentation",
        license="custom-academic",
        license_url="https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm",
        source_url="https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm",
        citation="Chiu et al., Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema, Biomedical Optics Express 2015.",
        paper_url="https://doi.org/10.1364/BOE.6.001172",
        bibtex=r"""@article{chiu2015duke,
    title   = {Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema},
    author  = {Chiu, Stephanie J. and Allingham, Michael J. and Mettu, Priyatham S. and Cousins, Scott W. and Izatt, Joseph A. and Farsiu, Sina},
    journal = {Biomedical Optics Express},
    volume  = {6},
    number  = {4},
    pages   = {1172--1194},
    year    = {2015},
    doi     = {10.1364/BOE.6.001172},
}""",
        notes="Random split with seed=42 (~64/16/20).",
    ),
    "JSRT": DatasetInfo(
        name="JSRT",
        modality="chest_xray",
        task="multilabel_segmentation",
        license="custom-academic",
        license_url="http://db.jsrt.or.jp/eng.php",
        source_url="http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/segmentation02.zip",
        citation="Shiraishi et al., Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule, AJR 2000.",
        paper_url="https://doi.org/10.2214/ajr.174.1.1740071",
        bibtex=r"""@article{shiraishi2000jsrt,
    title   = {Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule: Receiver Operating Characteristic Analysis of Radiologists' Detection of Pulmonary Nodules},
    author  = {Shiraishi, Junji and Katsuragawa, Shigehiko and Ikezoe, Junpei and Matsumoto, Tsuneo and Kobayashi, Takeshi and Komatsu, Ken-ichi and Matsui, Mitate and Fujita, Hiroshi and Kodera, Yoshie and Doi, Kunio},
    journal = {American Journal of Roentgenology},
    volume  = {174},
    number  = {1},
    pages   = {71--74},
    year    = {2000},
    doi     = {10.2214/ajr.174.1.1740071},
}""",
        notes="Uses the upstream official train/test split; train is sliced 90/10 into train/val by index (no random seed needed).",
    ),
    "PAXRay": DatasetInfo(
        name="PAXRay",
        modality="chest_xray",
        task="multilabel_segmentation",
        license="CC-BY-NC-SA-4.0",
        license_url="https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation/blob/main/LICENSE.txt",
        source_url="https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation",
        citation="Seibold et al., Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding, BMVC 2022.",
        paper_url="https://arxiv.org/abs/2210.03416",
        bibtex=r"""@inproceedings{seibold2022paxray,
    title     = {Detailed Annotations of Chest {X-Rays} via {CT} Projection for Report Understanding},
    author    = {Seibold, Constantin and Rei{\ss}, Simon and Sarfraz, M. Saquib and Stiefelhagen, Rainer and Kleesiek, Jens},
    booktitle = {Proceedings of the 33rd British Machine Vision Conference (BMVC)},
    year      = {2022},
    eprint    = {2210.03416},
    archivePrefix = {arXiv},
}""",
        notes="166-class multilabel chest X-ray anatomy; splits JSON ships with the upstream download.",
    ),
    "PAXRay++": DatasetInfo(
        name="PAXRay++",
        modality="chest_xray",
        task="multilabel_segmentation",
        license="CC-BY-NC-SA-4.0",
        license_url="https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation/blob/main/LICENSE.txt",
        source_url="https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation",
        citation="Seibold et al., Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling, 2023.",
        paper_url="https://arxiv.org/abs/2306.03934",
        bibtex=r"""@article{seibold2023paxraypp,
    title         = {Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling},
    author        = {Seibold, Constantin and Rei{\ss}, Simon and Sarfraz, Saquib and Fink, Matthias A. and Mayer, Victoria and Sellner, Jan and Kim, Moon Sung and Maier-Hein, Klaus H. and Kleesiek, Jens and Stiefelhagen, Rainer},
    journal       = {arXiv preprint arXiv:2306.03934},
    year          = {2023},
    eprint        = {2306.03934},
    archivePrefix = {arXiv},
}""",
        notes="Extended PAXRay variant.",
    ),
    "Teeth": DatasetInfo(
        name="Teeth",
        modality="panoramic_xray",
        task="detection",
        license="custom-academic",
        license_url="https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images",
        source_url="https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images",
        citation="Humans in the Loop, Teeth Segmentation on Dental X-Ray Images (Kaggle).",
        paper_url=None,
        bibtex=r"""@misc{hitl2023teeth,
    title        = {Teeth Segmentation on Dental {X-Ray} Images},
    author       = {{Humans in the Loop}},
    year         = {2023},
    howpublished = {Kaggle dataset},
    url          = {https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images},
}""",
        notes="Polygon annotations converted to COCO format. Random split with seed=42 (70/10/20).",
    ),
    "BS80k": DatasetInfo(
        name="BS80k",
        modality="scintigraphy",
        task="multilabel_segmentation",
        license="unknown",
        license_url=None,
        source_url="https://pubmed.ncbi.nlm.nih.gov/36334360/",
        citation="Huang et al., BS-80K: The first large open-access dataset of bone scan images, Computers in Biology and Medicine 2022.",
        paper_url="https://doi.org/10.1016/j.compbiomed.2022.106221",
        bibtex=r"""@article{huang2022bs80k,
    title   = {{BS-80K}: The first large open-access dataset of bone scan images},
    author  = {Huang, Zongmo and Pu, Xiaorong and Tang, Gongshun and Ping, Ming and Jiang, Guo and Wang, Mengjie and Wei, Xiaoyu and Ren, Yazhou},
    journal = {Computers in Biology and Medicine},
    volume  = {151},
    pages   = {106221},
    year    = {2022},
    doi     = {10.1016/j.compbiomed.2022.106221},
}""",
        notes="82,544 bone scan images / 3,247 patients from West China Hospital. Hosted on Google Drive by the authors; no explicit dataset license — treat as research-use-only and contact authors for any redistribution.",
    ),
    "MedakaHeart": DatasetInfo(
        name="MedakaHeart",
        modality="microscopy",
        task="multiclass_segmentation",
        license="GPL-3.0",
        license_url="https://www.gnu.org/licenses/gpl-3.0.en.html",
        source_url="https://osf.io/uyk79/",
        citation="Schutera et al., Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels, PLOS ONE 2022.",
        paper_url="https://doi.org/10.1371/journal.pone.0263656",
        bibtex=r"""@article{schutera2022frugal,
    title   = {Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels},
    author  = {Schutera, Mark and Rettenberger, Luca and Pylatiuk, Christian and Reischl, Markus},
    journal = {PLOS ONE},
    volume  = {17},
    number  = {2},
    pages   = {e0263656},
    year    = {2022},
    doi     = {10.1371/journal.pone.0263656},
}""",
        notes="Raw data + labels redistributed via OSF project uyk79 under GPL-3.0. 565 annotated train frames; two unseen-specimen test sets pooled into the test split.",
    ),
    "JumpBroadcast": DatasetInfo(
        name="JumpBroadcast",
        modality="rgb_video",
        task="multiclass_segmentation",
        license="custom-academic",
        license_url="https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/",
        source_url="https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/",
        citation="Ludwig et al., 2023. All Keypoints You Need: Detecting Arbitrary Keypoints on the Body of Triple, High, and Long Jump Athletes. CVPRW.",
        paper_url="https://arxiv.org/abs/2304.02939",
        bibtex=r"""@inproceedings{ludwig2023all_kps,
    title     = {All Keypoints You Need: Detecting Arbitrary Keypoints on the Body of Triple, High, and Long Jump Athletes},
    author    = {Ludwig, Katja and Lorenz, Julian and Sch{\"o}n, Robin and Lienhart, Rainer},
    booktitle = {Proceedings of the 2023 IEEE/CVF International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    month     = {June},
    year      = {2023},
}""",
        notes="Upstream test annotations gated; we re-stratify athletes 70/15/15 with seed=42 so the public release has a usable test split.",
    ),
    "Montgomery": DatasetInfo(
        name="Montgomery",
        modality="chest_xray",
        task="multiclass_segmentation",
        license="custom-academic",
        license_url="https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html",
        source_url="https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
        citation="Jaeger et al., Two public chest X-ray datasets for computer-aided screening of pulmonary diseases, Quantitative Imaging in Medicine and Surgery 2014.",
        paper_url="https://doi.org/10.3978/j.issn.2223-4292.2014.11.20",
        bibtex=r"""@article{jaeger2014montgomery,
    title   = {Two public chest {X-ray} datasets for computer-aided screening of pulmonary diseases},
    author  = {Jaeger, Stefan and Candemir, Sema and Antani, Sameer and W{\'a}ng, Yi-Xi{\'a}ng J. and Lu, Pu-Xuan and Thoma, George},
    journal = {Quantitative Imaging in Medicine and Surgery},
    volume  = {4},
    number  = {6},
    pages   = {475--477},
    year    = {2014},
    doi     = {10.3978/j.issn.2223-4292.2014.11.20},
}""",
        notes=(
            "138 PA chest radiographs (80 normal, 58 TB) with bilateral lung "
            "masks shipped as ManualMask/leftMask + rightMask. No official "
            "split; random 80/10/10 with seed=42, stratified on the TB/normal "
            "filename flag. NLM file naming is image-side; we re-map to "
            "anatomical class IDs so labels match the rest of the chest X-ray "
            "datasets in this repo: 0=background, 1=left_lung (patient's left, "
            "viewer's right), 2=right_lung (patient's right, viewer's left). "
            "The upstream ReadMe carries a redistribution clause ('do not "
            "share the data set outside of your research group/organization'), "
            "so we only ship the download script, never the data."
        ),
    ),
    "FIVES": DatasetInfo(
        name="FIVES",
        modality="fundus",
        task="multiclass_segmentation",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source_url="https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169",
        citation="Jin et al., FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation, Scientific Data 9, 475 (2022).",
        paper_url="https://doi.org/10.1038/s41597-022-01564-3",
        bibtex=r"""@article{jin2022fives,
    title   = {{FIVES}: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation},
    author  = {Jin, Kai and Huang, Xingru and Zhou, Jingxin and Li, Yunxiang and Yan, Yan and Sun, Yibao and Zhang, Qianni and Wang, Yaqi and Ye, Juan},
    journal = {Scientific Data},
    volume  = {9},
    number  = {1},
    pages   = {475},
    year    = {2022},
    doi     = {10.1038/s41597-022-01564-3},
}""",
        notes=(
            "800 high-resolution color fundus images (600 train / 200 test "
            "official split) balanced across Normal / Age-related macular "
            "degeneration / Diabetic retinopathy / Glaucoma. Binary vessel "
            "annotations curated via expert crowdsourcing. We keep the "
            "upstream train/test partition and carve a 80/20 train/val out of "
            "the train split with seed=42 stratified on the disease flag. "
            "Upstream ships as a single ~1.76 GB RAR5 archive on Figshare; "
            "extraction requires `unar` (`brew install unar` on macOS) or "
            "`unrar`. The default macOS `7z` from Homebrew p7zip does not "
            "support RAR5 and will fail."
        ),
    ),
    "HRF": DatasetInfo(
        name="HRF",
        modality="fundus",
        task="multiclass_segmentation",
        license="CC-BY-4.0",
        license_url="https://www5.cs.fau.de/research/data/fundus-images/",
        source_url="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip",
        citation="Budai et al., Robust Vessel Segmentation in Fundus Images, International Journal of Biomedical Imaging 2013.",
        paper_url="https://doi.org/10.1155/2013/154860",
        bibtex=r"""@article{budai2013hrf,
    title   = {Robust Vessel Segmentation in Fundus Images},
    author  = {Budai, Attila and Bock, R{\"u}diger and Maier, Andreas and Hornegger, Joachim and Michelson, Georg},
    journal = {International Journal of Biomedical Imaging},
    volume  = {2013},
    pages   = {154860},
    year    = {2013},
    doi     = {10.1155/2013/154860},
}""",
        notes=(
            "45 high-resolution fundus images (3504x2336 or 3072x2048): "
            "15 healthy, 15 glaucoma, 15 diabetic retinopathy. Binary vessel "
            "annotations (manual1/*.tif) and FOV ROI masks (mask/*_mask.tif). "
            "No official split; random 70/15/15 with seed=42, stratified on "
            "the condition flag (h/g/dr) so each split keeps the original "
            "case-mix ratio. Labels: 0=background, 1=vessel."
        ),
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    """Look up dataset metadata by name. Raises KeyError if unknown."""
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Known: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name]
