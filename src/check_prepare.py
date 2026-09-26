#confirms whehter the cleaning has been worked & have we got the clean data?
"""Checks after Stage 1: similarity of true pairs + sample cleaned rows per country.

    python -m src.check_prepare --work work
"""
import argparse
from pathlib import Path

import polars as pl

from . import config

pl.Config.set_fmt_str_lengths(70)
pl.Config.set_tbl_width_chars(220)

COLS = ["rid", "src", "country", "name_core", "name_compact", "addr_clean", "addr_numbers"]


def tokens(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ").list.unique()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=config.DEFAULT_WORK_DIR)
    ap.add_argument("--sample", type=int, default=300_000)
    args = ap.parse_args()
    w = args.work

    pairs = pl.read_parquet(config.pairs_path(w)).sample(args.sample, seed=config.SEED)
    s1 = pl.read_parquet(config.clean_path(w, "train", 1), columns=COLS)
    s23 = pl.concat([pl.read_parquet(config.clean_path(w, "train", s), columns=COLS) for s in (2, 3)])
    p = (
        pairs.join(s1.drop("src").rename(lambda c: c + "_1"), left_on="s1_rid", right_on="rid_1")
        .join(s23.rename(lambda c: c + "_2"), left_on=["src", "cand_rid"], right_on=["src_2", "rid_2"])
    )
    stats = p.select(
        (pl.col("name_core_1") == pl.col("name_core_2")).mean().alias("name_core_equal"),
        (pl.col("name_compact_1") == pl.col("name_compact_2")).mean().alias("name_compact_equal"),
        (tokens("name_core_1").list.set_intersection(tokens("name_core_2")).list.len() > 0)
        .mean().alias("share_name_token"),
        (pl.col("addr_numbers_1").list.set_intersection(pl.col("addr_numbers_2")).list.len() > 0)
        .mean().alias("share_addr_number"),
        (tokens("addr_clean_1").list.set_intersection(tokens("addr_clean_2")).list.len() > 0)
        .mean().alias("share_addr_token"),
    )
    print("True-pair similarity after cleaning:")
    for k, v in stats.row(0, named=True).items():
        print(f"  {k:22s} {v:.1%}")

    t1 = pl.read_parquet(config.clean_path(w, "test", 1))
    for c in t1["country"].unique().sort():
        print(f"\nTest S1 sample — {c}")
        print(t1.filter(pl.col("country") == c).sample(5, seed=1)
              .select("business_name", "name_core", "legal_form", "addr_clean"))


if __name__ == "__main__":
    main()