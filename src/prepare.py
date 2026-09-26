"""Stages 0-1: raw TSVs -> cleaned Parquet files.

    python -m src.prepare --data ../../student_resource/dataset --work work
"""
import argparse
import time
from pathlib import Path

import polars as pl

from . import config
from .ingest import read_ground_truth_pairs, read_source
from .normalize import load_indic, normalize

KEEP = [
    "rid", "entity_id", "src", "country", "business_name", "business_address",
    "name_clean", "name_core", "name_compact", "name_skel", "legal_form", "is_web", "name_nonlatin",
    "addr_clean", "addr_numbers", "postcode", "is_landmark", "addr_missing",
]
CHUNK = 500_000


def prepare_split(data_dir: Path, work_dir: Path, split: str) -> None:
    for src in config.SOURCES:
        t = time.time()
        raw = read_source(data_dir, split, src)
        parts = [normalize(raw.slice(i, CHUNK)).select(KEEP) for i in range(0, raw.height, CHUNK)]
        del raw
        df = pl.concat(parts)
        del parts
        out = config.clean_path(work_dir, split, src)
        df.write_parquet(out)
        counts = dict(df["country"].value_counts().iter_rows())
        print(f"  {split} S{src}: {df.height:,} rows {counts} -> {out.name} ({time.time()-t:.0f}s)")


def prepare_pairs(data_dir: Path, work_dir: Path) -> None:
    pairs = read_ground_truth_pairs(data_dir)
    s1 = pl.read_parquet(config.clean_path(work_dir, "train", 1), columns=["entity_id", "rid"])
    s23 = pl.concat([
        pl.read_parquet(config.clean_path(work_dir, "train", s), columns=["entity_id", "rid", "src"])
        for s in (2, 3)
    ])
    out = (
        pairs.join(s1.rename({"entity_id": "s1_id", "rid": "s1_rid"}), on="s1_id", how="inner")
        .join(s23.rename({"entity_id": "cand_id", "rid": "cand_rid"}), on="cand_id", how="inner")
        .select("s1_rid", "src", "cand_rid")
    )
    if out.height != pairs.height:
        raise ValueError(f"lost {pairs.height - out.height} ground-truth pairs while joining ids")
    out.write_parquet(config.pairs_path(work_dir))
    print(f"  train pairs: {out.height:,} -> {config.pairs_path(work_dir).name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.DEFAULT_DATA_DIR)
    ap.add_argument("--work", type=Path, default=config.DEFAULT_WORK_DIR)
    ap.add_argument("--splits", nargs="+", default=list(config.SPLITS))
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    n = load_indic(args.work / "indic_dict.json")
    print(f"[prepare] Indian-script dictionary: {n} words" + ("" if n else " (none found - run src.build_dict first)"))
    for split in args.splits:
        print(f"[prepare] {split}")
        prepare_split(args.data, args.work, split)
    if "train" in args.splits:
        prepare_pairs(args.data, args.work)


if __name__ == "__main__":
    main()