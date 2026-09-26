"""v4 features on top of features.py.

Step 4 (error-analysis features, added to every pair):
  cos_name, cos_addr, cos_comb  TF-IDF cosines from the similarity search
  s1_name_count   how common this S1's compact name is among S1 of the same country
                  (per million records, so it means the same thing in every country)
  s23_name_count  how common the candidate's compact name is among S2/S3 (per million)
                  -> a very common name ("sai enterprises") is weak evidence on its own
  num_all_equal   both records have the same set of address numbers
  num_last_equal  last address number equal (catches decoys like W-12/12 vs W-12/15)
  num_b_subset_a  every number of the candidate also appears in the S1 address
  ce, ce_gap, ce_rank   cross-encoder score (ce.py), gap to this S1's best score, rank within the S1

Step 5 (cross-source agreement, second stage). Needs a first-stage probability p1 per pair:
  A real business usually has 3-4 records spread over S2 AND S3, all noisy copies of the same
  thing, so they look like each other. Decoys are one-off near-copies. For each pair we add:
  p1                       first-stage probability
  p1_rank / p1_sum / n_strong   rank of p1 within the S1, sum of p1, number of candidates with p1>0.5
  p1_best_other_src        p1 of the S1's best candidate in the OTHER source
  sim_name_other / sim_addr_other   similarity of this candidate to that best other-source candidate
  p1_owner_gap             best p1 any S1 gives this record minus this S1's p1 (competition)
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .features import PAIR_COLS, CONTEXT_COLS

V4_EXTRA = ["cos_name", "cos_addr", "cos_comb", "s1_name_count", "s23_name_count",
            "num_all_equal", "num_last_equal", "num_b_subset_a",
            "ce", "ce_gap", "ce_rank"]   # cross-encoder score (ce.py), its gap to the S1's best, its rank
FEATURES_V4 = ["src"] + CONTEXT_COLS + PAIR_COLS + V4_EXTRA
STAGE2_EXTRA = ["p1", "p1_rank", "p1_sum", "n_strong", "p1_best_other_src",
                "sim_name_other", "sim_addr_other", "p1_owner_gap"]
FEATURES_STAGE2 = FEATURES_V4 + STAGE2_EXTRA

PRUNE_FEATURES = ["src", "cos_name", "cos_addr", "cos_comb", "score", "rank_in_s1",
                  "gap_to_best_in_s1", "n_cands_s1", "rank_in_s1_src", "rank_in_cand",
                  "gap_to_best_in_cand", "n_s1_for_cand", "rank_comb_in_s1"]


def name_counts(c1: pl.DataFrame, c23: pl.DataFrame):
    """Lookup tables: how often each compact name appears among S1 and among S2/S3,
    as a rate per million records of that country (so a country with more records does
    not automatically look like it has more 'common' names; France only exists in test)."""
    a = c1.group_by("name_compact").len().select(
        "name_compact", (pl.col("len") * (1e6 / max(c1.height, 1))).alias("s1_name_count"))
    b = c23.group_by("name_compact").len().select(
        "name_compact", (pl.col("len") * (1e6 / max(c23.height, 1))).alias("s23_name_count"))
    return a, b


def add_v4_extras(feats: pl.DataFrame, c1: pl.DataFrame, c23: pl.DataFrame, counts) -> pl.DataFrame:
    a, b = counts
    r1 = c1.select(pl.col("rid").alias("s1_rid"), pl.col("name_compact").alias("_n1"), pl.col("addr_numbers").alias("_x1"))
    r2 = c23.select("src", pl.col("rid").alias("cand_rid"), pl.col("name_compact").alias("_n2"), pl.col("addr_numbers").alias("_x2"))
    f = feats.join(r1, on="s1_rid", how="left").join(r2, on=["src", "cand_rid"], how="left")
    f = f.join(a, left_on="_n1", right_on="name_compact", how="left").join(b, left_on="_n2", right_on="name_compact", how="left")
    x1, x2 = pl.col("_x1"), pl.col("_x2")
    both = (x1.list.len() > 0) & (x2.list.len() > 0)
    f = f.with_columns(
        pl.when(both).then((x1.list.sort() == x2.list.sort()).cast(pl.Float32)).alias("num_all_equal"),
        pl.when(both).then((x1.list.last() == x2.list.last()).cast(pl.Float32)).alias("num_last_equal"),
        pl.when(both).then((x2.list.set_difference(x1).list.len() == 0).cast(pl.Float32)).alias("num_b_subset_a"),
        pl.col("s1_name_count").fill_null(0).cast(pl.Float32),
        pl.col("s23_name_count").fill_null(0).cast(pl.Float32),
    )
    return f.drop("_n1", "_n2", "_x1", "_x2")


def add_stage2(df: pl.DataFrame, c23: pl.DataFrame) -> pl.DataFrame:
    """df: pairs with p1. Adds the cross-source agreement features (see module docstring)."""
    df = df.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1_rid").cast(pl.Float32).alias("p1_rank"),
        pl.col("p1").sum().over("s1_rid").alias("p1_sum"),
        (pl.col("p1") > 0.5).sum().over("s1_rid").cast(pl.Float32).alias("n_strong"),
        (pl.col("p1").max().over("src", "cand_rid") - pl.col("p1")).alias("p1_owner_gap"),
    )
    # best candidate per (S1, source)
    best = (df.sort("p1", descending=True).unique(subset=["s1_rid", "src"], keep="first")
            .select("s1_rid", pl.col("src").alias("other_src"), pl.col("cand_rid").alias("other_rid"),
                    pl.col("p1").alias("p1_best_other_src")))
    df = df.with_columns((5 - pl.col("src")).alias("other_src"))  # 2 <-> 3
    df = df.join(best, on=["s1_rid", "other_src"], how="left")
    rec = c23.select("src", "rid", "name_core", "addr_clean")
    df = (df.join(rec.rename({"src": "src_a", "rid": "rid_a", "name_core": "_na", "addr_clean": "_aa"}),
                  left_on=["src", "cand_rid"], right_on=["src_a", "rid_a"], how="left")
            .join(rec.rename({"src": "src_b", "rid": "rid_b", "name_core": "_nb", "addr_clean": "_ab"}),
                  left_on=["other_src", "other_rid"], right_on=["src_b", "rid_b"], how="left"))
    has = df["_nb"].is_not_null().to_numpy()
    na, nb = df["_na"].fill_null("").to_list(), df["_nb"].fill_null("").to_list()
    aa, ab = df["_aa"].fill_null("").to_list(), df["_ab"].fill_null("").to_list()
    sn = cpdist(na, nb, scorer=fuzz.token_set_ratio, workers=-1)
    sa = cpdist(aa, ab, scorer=fuzz.token_set_ratio, workers=-1)
    df = df.with_columns(
        pl.Series("sim_name_other", np.where(has, sn, np.nan).astype(np.float32)),
        pl.Series("sim_addr_other", np.where(has, sa, np.nan).astype(np.float32)),
    )
    return df.drop("other_src", "other_rid", "_na", "_nb", "_aa", "_ab")