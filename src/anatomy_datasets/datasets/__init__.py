"""Per-dataset Dataset classes.

Each module defines one or more concrete ``*_Dataset`` classes that are
thin aliases / subclasses of the three base classes in
``anatomy_datasets.base``.
"""

from .bs80k import BS80KAnatomy_detection_Dataset
from .chasedb1 import CHASEDB1_Dataset
from .drive import DRIVE_Dataset
from .duke import DUKE_Dataset
from .jsrt import JSRT_binary_Dataset, JSRT_detection_Dataset
from .jump_broadcast import JumpBroadcast_Dataset
from .medaka_heart import MedakaHeart_Dataset
from .fives import FIVES_Dataset
from .hrf import HRF_Dataset
from .lesav import (
    LESAV_AV_Dataset,
    LESAV_AV_detection_Dataset,
    LESAV_Vessel_Dataset,
    LESAV_Vessel_detection_Dataset,
)
from .montgomery import Montgomery_Dataset
from .paxray import (
    PAXRay4_binary_Dataset,
    PAXRay4_detection_Dataset,
    PAXRay166_binary_Dataset,
    PAXRay166_detection_Dataset,
)
from .paxraypp import PAXRayPP_binary_Dataset, PAXRayPP_Instance_Dataset
from .ravir import RAVIR_Dataset
from .stare import STARE_Dataset
from .teeth import Teeth_Dataset

__all__ = [
    "BS80KAnatomy_detection_Dataset",
    "CHASEDB1_Dataset",
    "DRIVE_Dataset",
    "DUKE_Dataset",
    "JSRT_binary_Dataset",
    "JSRT_detection_Dataset",
    "JumpBroadcast_Dataset",
    "MedakaHeart_Dataset",
    "FIVES_Dataset",
    "HRF_Dataset",
    "LESAV_AV_Dataset",
    "LESAV_AV_detection_Dataset",
    "LESAV_Vessel_Dataset",
    "LESAV_Vessel_detection_Dataset",
    "Montgomery_Dataset",
    "PAXRay4_binary_Dataset",
    "PAXRay4_detection_Dataset",
    "PAXRay166_binary_Dataset",
    "PAXRay166_detection_Dataset",
    "PAXRayPP_binary_Dataset",
    "PAXRayPP_Instance_Dataset",
    "RAVIR_Dataset",
    "STARE_Dataset",
    "Teeth_Dataset",
]
