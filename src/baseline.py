# This runs the rule-based matcher. --mode tune tests it on training data and prints the score at each threshold. --mode predict writes the two output files for the test set

"""Baseline (upload #1): simple blocking keys + string-similarity rule + one-owner rule.

    python -m src.baseline --mode tune                      # score on a train sample
    python -m src.baseline --mode predict --threshold 85    # write output/ files for test
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import config
from .evaluate import macro_f05

COLS = ["rid", "src", "country", "entity_id", "name_core", "name_compact", "addr_clean", "addr_numbers"]
MAX_BLOCK = 30      # skip keys shared by more than this many S2/S3 records (too common)
MAX_CANDS = 40      # keep at most this many candidates per S1
CHUNK = 200_000     # S1 rows scored at a time (keeps memory under ~4 GB)


def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Three cheap blocking keys per record (null when not usable)."""
    first_num = pl.col("addr_numbers").list.first()
    words = pl.col("name_core").str.split(" ")
    return df.with_columns(
        pl.when(pl.col("name_compact").str.len_chars() >= 4)
        .then(pl.col("name_compact")).alias("k_name"),
        pl.when(first_num.is_not_null() & (words.list.first().str.len_chars() >= 3))
        .then(first_num + "|" + words.list.first()).alias("k_num_word"),
        pl.when(words.list.len() >= 2)
        .then(words.list.slice(0, 2).list.join(" ")).alias("k_two_words"),
    )


def candidates(s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """All (s1_rid, src, cand_rid) pairs sharing at least one key within the same country."""
    s1k, s23k = add_keys(s1), add_keys(s23)
    parts = []
    for key in ("k_name", "k_num_word", "k_two_words"):
        right = s23k.select("country", key, "src", "rid").drop_nulls(key)
        sizes = right.group_by("country", key).len()
        right = right.join(sizes.filter(pl.col("len") <= MAX_BLOCK), on=["country", key]).drop("len")
        left = s1k.select("country", key, pl.col("rid").alias("s1_rid")).drop_nulls(key)
        parts.append(left.join(right, on=["country", key]).select("s1_rid", "src", pl.col("rid").alias("cand_rid")))
    return pl.concat(parts).unique()


def score(pairs: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """Attach name/address similarity (0-100) and a combined score."""
    p = (
        pairs.join(s1.select(pl.col("rid").alias("s1_rid"), pl.col("name_core").alias("n1"),
                             pl.col("addr_clean").alias("a1")), on="s1_rid")
        .join(s23.select("src", pl.col("rid").alias("cand_rid"), pl.col("name_core").alias("n2"),
                         pl.col("addr_clean").alias("a2")), on=["src", "cand_rid"])
    )
    name_sim = cpdist(p["n1"].to_list(), p["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    addr_sim = cpdist(p["a1"].to_list(), p["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    addr_missing = (p["a2"] == "").to_numpy()
    combined = np.where(addr_missing, name_sim, 0.6 * name_sim + 0.4 * addr_sim)
    p = p.with_columns(pl.Series("score", combined.astype(np.float32))).drop("n1", "n2", "a1", "a2")
    # keep the best MAX_CANDS per S1 (this is the set written to candidate_pairs.tsv)
    return p.sort("score", descending=True).group_by("s1_rid").head(MAX_CANDS)


def score_in_chunks(s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """Same as score(candidates(...)) but one country and one S1 chunk at a time."""
    out = []
    for country in s1["country"].unique().to_list():   # open set: whatever countries exist
        c1 = s1.filter(pl.col("country") == country)
        c23 = s23.filter(pl.col("country") == country)
        for i in range(0, c1.height, CHUNK):
            part = c1.slice(i, CHUNK)
            out.append(score(candidates(part, c23), part, c23).select("s1_rid", "src", "cand_rid", "score"))
        print(f"  scored {country}: {c1.height:,} S1")
    return pl.concat(out)


def assign(scored: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """One owner per S2/S3 record (its best S1), then keep pairs above the threshold."""
    best = scored.sort("score", descending=True).unique(subset=["src", "cand_rid"], keep="first")
    return best.filter(pl.col("score") >= threshold)


def load(work: Path, split: str):
    s1 = pl.read_parquet(config.clean_path(work, split, 1), columns=COLS)
    s23 = pl.concat([pl.read_parquet(config.clean_path(work, split, s), columns=COLS) for s in (2, 3)])
    return s1, s23


def to_lists(pairs: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame, col: str) -> pl.DataFrame:
    """(s1_rid, src, cand_rid) pairs -> one row per S1 with comma-joined entity ids.
    Every S1 gets a row; S1s with no pairs get an empty string."""
    ids = pairs.join(s23.select("src", pl.col("rid").alias("cand_rid"), pl.col("entity_id").alias("cid")),
                     on=["src", "cand_rid"])
    lists = ids.group_by("s1_rid").agg(pl.col("cid").unique().sort().str.join(",").alias(col))
    return (
        s1.select(pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("source1_entity_id"))
        .join(lists, on="s1_rid", how="left")
        .select("source1_entity_id", pl.col(col).fill_null(""))
    )


def tune(work: Path, n_s1: int) -> None:
    t = time.time()
    s1, s23 = load(work, "train")
    s1 = s1.sample(n_s1, seed=config.SEED)
    truth_pairs = pl.read_parquet(config.pairs_path(work)).join(
        s1.select(pl.col("rid").alias("s1_rid")), on="s1_rid")
    scored = score(candidates(s1, s23), s1, s23)
    hits = scored.join(truth_pairs, on=["s1_rid", "src", "cand_rid"]).height
    print(f"sample S1={n_s1:,}  candidates={scored.height:,} ({scored.height/n_s1:.1f}/S1)  "
          f"pair recall={hits/truth_pairs.height:.1%}  ({time.time()-t:.0f}s)")

    truth = {r: set() for r in s1["rid"].to_list()}
    for a, s, c in truth_pairs.iter_rows():
        truth[a].add((s, c))
    for th in (60, 65, 70, 75, 80, 85, 90, 95):
        pred = {}
        for a, s, c in assign(scored, th).select("s1_rid", "src", "cand_rid").iter_rows():
            pred.setdefault(a, set()).add((s, c))
        print(f"  threshold {th}: macro F0.5 = {macro_f05(truth, pred):.4f}")


def predict(work: Path, out_dir: Path, threshold: float) -> None:
    t = time.time()
    s1, s23 = load(work, "test")
    scored = score_in_chunks(s1, s23)
    matched = assign(scored, threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    # quote_style="never" so empty lists are written as blank, not ""
    to_lists(scored, s1, s23, "candidate_entity_ids").write_csv(
        out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    to_lists(matched, s1, s23, "matched_entity_ids").write_csv(
        out_dir / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"test S1={s1.height:,} candidates={scored.height:,} matches={matched.height:,} "
          f"-> {out_dir} ({time.time()-t:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "predict"], required=True)
    ap.add_argument("--work", type=Path, default=config.DEFAULT_WORK_DIR)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--threshold", type=float, default=85)
    ap.add_argument("--sample", type=int, default=200_000)
    args = ap.parse_args()
    if args.mode == "tune":
        tune(args.work, args.sample)
    else:
        predict(args.work, args.out, args.threshold)


if __name__ == "__main__":
    main()