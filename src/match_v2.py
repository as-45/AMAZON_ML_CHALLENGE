# This gives each candidate a quick score (60% name + 40% address). The model uses it as a feature and reuses its countries() and score() functions.
"""Upload #2: v2 blocking (blocking.py) + similarity rule + one-owner rule.

    python -m src.match_v2 --mode tune                     # recall + F0.5 on a train sample
    python -m src.match_v2 --mode predict --threshold 85   # write output/ files for test
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import blocking, config
from .baseline import assign, to_lists
from .evaluate import macro_f05

COLS = ["rid", "src", "country", "entity_id", "name_core", "name_compact", "name_skel",
        "addr_clean", "addr_numbers"]
MAX_CANDS = 40      # keep at most this many candidates per S1 (best by score)
S1_CHUNK = 200_000  # S1 rows handled at a time


def score(pairs: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """Name similarity = best of plain-name and skeleton-name token_set_ratio (0-100).
    Combined score = 0.6 * name + 0.4 * address (name only if the address is missing)."""
    p = (
        pairs.join(s1.select(pl.col("rid").alias("s1_rid"), pl.col("name_core").alias("n1"),
                             pl.col("name_skel").alias("k1"), pl.col("addr_clean").alias("a1")), on="s1_rid")
        .join(s23.select("src", pl.col("rid").alias("cand_rid"), pl.col("name_core").alias("n2"),
                         pl.col("name_skel").alias("k2"), pl.col("addr_clean").alias("a2")),
              on=["src", "cand_rid"])
    )
    sim = lambda a, b: cpdist(p[a].to_list(), p[b].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    name_sim = np.maximum(sim("n1", "n2"), sim("k1", "k2"))
    addr_sim = sim("a1", "a2")
    addr_missing = (p["a2"] == "").to_numpy()
    combined = np.where(addr_missing, name_sim, 0.6 * name_sim + 0.4 * addr_sim)
    p = p.select("s1_rid", "src", "cand_rid").with_columns(pl.Series("score", combined.astype(np.float32)))
    return p.sort("score", "src", "cand_rid", descending=[True, False, False]).group_by("s1_rid").head(MAX_CANDS)


def load_country(work: Path, split: str, country: str, s1_rids=None):
    """Read only one country's records from disk (keeps memory low)."""
    scan = lambda s: pl.scan_parquet(config.clean_path(work, split, s)).select(COLS).filter(
        pl.col("country") == country)
    c1 = scan(1).collect()
    if s1_rids is not None:
        c1 = c1.join(s1_rids, on="rid", how="semi")
    c23 = pl.concat([scan(2).collect(), scan(3).collect()])
    return c1, c23


def countries(work: Path, split: str) -> list:
    return sorted(pl.read_parquet(config.clean_path(work, split, 1), columns=["country"])
                  ["country"].unique().to_list())


def block_and_score(work: Path, split: str, s1_rids=None) -> pl.DataFrame:
    """Blocking + scoring, one country at a time (the country list comes from the data)."""
    out = []
    for country in countries(work, split):
        t = time.time()
        c1, c23 = load_country(work, split, country, s1_rids)
        dfs = blocking.word_df(c23)
        c23k = blocking.add_keys(c23, dfs)
        part_keys = blocking.add_keys(c1, dfs)
        del dfs
        c23 = c23.select("rid", "src", "name_core", "name_skel", "addr_clean")  # only what score() needs
        for i in range(0, c1.height, S1_CHUNK):
            part = c1.slice(i, S1_CHUNK)
            cand = blocking.candidates(part_keys.join(part.select("rid"), on="rid", how="semi"), c23k)
            out.append(score(cand, part, c23))
        print(f"  {country}: {c1.height:,} S1 blocked + scored ({time.time()-t:.0f}s)")
        del c1, c23, c23k, part_keys
    return pl.concat(out)


def tune(work: Path, n_per_country: int) -> None:
    s1 = pl.read_parquet(config.clean_path(work, "train", 1), columns=["rid", "country"])
    s1 = pl.concat([
        s1.filter(pl.col("country") == c).sample(n_per_country, seed=config.SEED)
        for c in sorted(s1["country"].unique().to_list())
    ])
    truth_pairs = pl.read_parquet(config.pairs_path(work)).join(
        s1.select(pl.col("rid").alias("s1_rid")), on="s1_rid")
    scored = block_and_score(work, "train", s1.select("rid"))
    hits = scored.join(truth_pairs, on=["s1_rid", "src", "cand_rid"]).height
    print(f"sample S1={s1.height:,}  candidates={scored.height:,} ({scored.height/s1.height:.1f}/S1)  "
          f"blocking recall={hits/truth_pairs.height:.1%}")
    truth = {r: set() for r in s1["rid"].to_list()}
    for a, s, c in truth_pairs.iter_rows():
        truth[a].add((s, c))
    for th in (70, 75, 80, 85, 90):
        pred = {}
        for a, s, c in assign(scored, th).select("s1_rid", "src", "cand_rid").iter_rows():
            pred.setdefault(a, set()).add((s, c))
        print(f"  threshold {th}: macro F0.5 = {macro_f05(truth, pred):.4f}")


def predict(work: Path, out_dir: Path, threshold: float) -> None:
    t = time.time()
    scored = block_and_score(work, "test")
    matched = assign(scored, threshold)
    ids = ["rid", "src", "entity_id"]
    s1 = pl.read_parquet(config.clean_path(work, "test", 1), columns=ids)
    s23 = pl.concat([pl.read_parquet(config.clean_path(work, "test", s), columns=ids) for s in (2, 3)])
    out_dir.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--sample", type=int, default=50_000, help="S1 per country for tune")
    args = ap.parse_args()
    if args.mode == "tune":
        tune(args.work, args.sample)
    else:
        predict(args.work, args.out, args.threshold)


if __name__ == "__main__":
    main()