"""Stage 0: read the raw TSV files safely."""
from pathlib import Path

import polars as pl

from . import config

READ_OPTS = dict(separator="\t", quote_char=None, infer_schema_length=0, encoding="utf8")


def read_source(data_dir: Path, split: str, src: int) -> pl.DataFrame:
    df = pl.read_csv(config.source_path(data_dir, split, src), **READ_OPTS)
    expected = ["entity_id", "business_name", "business_address", "country"]
    if df.columns != expected:
        raise ValueError(f"{split} source{src}: unexpected columns {df.columns}")
    return df.with_columns(
        pl.lit(src, dtype=pl.Int8).alias("src"),
        pl.int_range(pl.len(), dtype=pl.Int32).alias("rid"),
        pl.col("business_name").fill_null(""),
        pl.col("country").fill_null("UNKNOWN").str.strip_chars(),
    )


def read_ground_truth_pairs(data_dir: Path) -> pl.DataFrame:
    """One row per true (source1, source2/3) pair."""
    gt = pl.read_csv(config.ground_truth_path(data_dir), **READ_OPTS)
    return (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"})
    )