"""Shared paths and settings for the entity-resolution pipeline."""
from pathlib import Path

SEED = 42

_HERE = Path(__file__).resolve()
# Folder that contains train/ and test/. On Modal the code sits at /root/src, which has fewer
# parent folders, so fall back to a safe default there (Modal always passes --work anyway).
DEFAULT_DATA_DIR = (_HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent) / "student_resource" / "dataset"
# Where cleaned Parquet files are written.
DEFAULT_WORK_DIR = _HERE.parents[1] / "work"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def source_path(data_dir: Path, split: str, src: int) -> Path:
    return Path(data_dir) / split / f"{split}_source{src}.tsv"


def ground_truth_path(data_dir: Path) -> Path:
    return Path(data_dir) / "train" / "train_ground_truth.tsv"


def clean_path(work_dir: Path, split: str, src: int) -> Path:
    return Path(work_dir) / f"{split}_s{src}.parquet"


def pairs_path(work_dir: Path) -> Path:
    return Path(work_dir) / "train_pairs.parquet"