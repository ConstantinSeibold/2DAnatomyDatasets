"""Map short dataset aliases (``DRIVE``, ``STARE``, ...) to underlying class +
env var + default splits-JSON filename.

This is the table consulted by the convenience constructors in
``anatomy_datasets.__init__``. It is intentionally separate from
``registry.py`` (which holds licence/citation/modality metadata) because
``registry`` is consumed by exporters and external tooling, while this
table is only about runtime instantiation defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

from anatomy_datasets.base import (
    BaseDetectionDataset,
    BaseMultiClassDataset,
    BaseMultiLabelDataset,
)
from anatomy_datasets.datasets import (
    BS80KAnatomy_detection_Dataset,
    CHASEDB1_Dataset,
    DRIVE_Dataset,
    DUKE_Dataset,
    JSRT_binary_Dataset,
    JSRT_detection_Dataset,
    JumpBroadcast_Dataset,
    MedakaHeart_Dataset,
    Montgomery_Dataset,
    PAXRay4_binary_Dataset,
    PAXRay4_detection_Dataset,
    PAXRay166_binary_Dataset,
    PAXRay166_detection_Dataset,
    PAXRayPP_binary_Dataset,
    PAXRayPP_Instance_Dataset,
    RAVIR_Dataset,
    STARE_Dataset,
    Teeth_Dataset,
)


@dataclass(frozen=True)
class DiscoveryEntry:
    """How to locate / instantiate a dataset given just its short alias."""

    cls: Type
    env_var: str
    default_json_name: Optional[str]   # relative to root; None means must pass explicitly
    task: str                          # "segmentation" or "detection" - selects constructor sig
    registry_name: str                 # key into DATASET_REGISTRY


DISCOVERY: dict[str, DiscoveryEntry] = {
    "DRIVE": DiscoveryEntry(
        cls=DRIVE_Dataset, env_var="DRIVE_ROOT_FOLDER",
        default_json_name="drive_splits.json", task="segmentation",
        registry_name="DRIVE",
    ),
    "STARE": DiscoveryEntry(
        cls=STARE_Dataset, env_var="STARE_ROOT_FOLDER",
        default_json_name="stare_splits.json", task="segmentation",
        registry_name="STARE",
    ),
    "CHASE_DB1": DiscoveryEntry(
        cls=CHASEDB1_Dataset, env_var="CHASEDB1_ROOT_FOLDER",
        default_json_name="chasedb1_splits.json", task="segmentation",
        registry_name="CHASE_DB1",
    ),
    "RAVIR": DiscoveryEntry(
        cls=RAVIR_Dataset, env_var="RAVIR_ROOT_FOLDER",
        default_json_name="ravir_splits.json", task="segmentation",
        registry_name="RAVIR",
    ),
    "DUKE": DiscoveryEntry(
        cls=DUKE_Dataset, env_var="DUKE_ROOT_PATH",
        default_json_name="duke_splits.json", task="segmentation",
        registry_name="DUKE",
    ),
    "MedakaHeart": DiscoveryEntry(
        cls=MedakaHeart_Dataset, env_var="MEDAKA_HEART_ROOT_FOLDER",
        default_json_name="medaka_heart_splits.json", task="segmentation",
        registry_name="MedakaHeart",
    ),
    "JumpBroadcast": DiscoveryEntry(
        cls=JumpBroadcast_Dataset, env_var="JUMP_BROADCAST_ROOT_FOLDER",
        default_json_name="jump_broadcast_splits.json", task="segmentation",
        registry_name="JumpBroadcast",
    ),
    "Montgomery": DiscoveryEntry(
        cls=Montgomery_Dataset, env_var="MONTGOMERY_ROOT_FOLDER",
        default_json_name="montgomery_splits.json", task="segmentation",
        registry_name="Montgomery",
    ),
    "JSRT": DiscoveryEntry(
        cls=JSRT_binary_Dataset, env_var="JSRT_ROOT_PATH",
        default_json_name=None,  # JSRT splits path comes from JSRT_JSON_PATH env var directly
        task="segmentation",
        registry_name="JSRT",
    ),
    "PAXRay166": DiscoveryEntry(
        cls=PAXRay166_binary_Dataset, env_var="PAXRAY_ROOT_PATH",
        default_json_name="paxray.json", task="segmentation",
        registry_name="PAXRay",
    ),
    "PAXRay4": DiscoveryEntry(
        cls=PAXRay4_binary_Dataset, env_var="PAXRAY_ROOT_PATH",
        default_json_name="paxray.json", task="segmentation",
        registry_name="PAXRay",
    ),
    "PAXRayPP": DiscoveryEntry(
        cls=PAXRayPP_binary_Dataset, env_var="PAXRAYPP_ROOT",
        default_json_name=None, task="segmentation",
        registry_name="PAXRay++",
    ),
    # Detection variants
    "JSRTDetection": DiscoveryEntry(
        cls=JSRT_detection_Dataset, env_var="JSRT_ROOT_PATH",
        default_json_name=None, task="detection",
        registry_name="JSRT",
    ),
    "PAXRay4Detection": DiscoveryEntry(
        cls=PAXRay4_detection_Dataset, env_var="PAXRAY_ROOT_PATH",
        default_json_name=None, task="detection",
        registry_name="PAXRay",
    ),
    "PAXRay166Detection": DiscoveryEntry(
        cls=PAXRay166_detection_Dataset, env_var="PAXRAY_ROOT_PATH",
        default_json_name=None, task="detection",
        registry_name="PAXRay",
    ),
    "PAXRayPPInstance": DiscoveryEntry(
        cls=PAXRayPP_Instance_Dataset, env_var="PAXRAYPP_ROOT",
        default_json_name=None, task="detection",
        registry_name="PAXRay++",
    ),
    "BS80KAnatomyDetection": DiscoveryEntry(
        cls=BS80KAnatomy_detection_Dataset, env_var="BS80K_ROOT",
        default_json_name=None, task="detection",
        registry_name="BS80k",
    ),
    "Teeth": DiscoveryEntry(
        cls=Teeth_Dataset, env_var="TEETH_ROOT",
        default_json_name=None, task="detection",
        registry_name="Teeth",
    ),
}
