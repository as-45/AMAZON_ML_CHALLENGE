# # What it does: turns each (business, candidate) pair into 33 numbers the model learns from. They come in three groups:
# Pair features (24): how alike the two records are.
# The name is compared 7 ways: token_set, token_sort, ratio, partial, skeleton, compact ratio and Jaro-Winkler.
# The address is compared 3 ways.
# Also: whether the house numbers match, whether the legal forms clash (Inc vs Pvt Ltd), and whether the name is written as a website or in non-Latin script.


"""Stage 3: turn each (S1, candidate) pair into numbers the model can learn from.

Three groups of features:
  pair features     how alike the two records are (names, addresses, numbers, legal form)
  context features  how this candidate compares with the S1's OTHER candidates
  reverse features  how this S1 compares with OTHER S1s that also want this candidate
Nothing uses the country value or raw words, so the model works the same on France.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

FEAT_COLS_RECORD = ["rid", "src", "name_core", "name_compact", "name_skel", "legal_form",
                    "is_web", "name_nonlatin", "addr_clean", "addr_numbers", "is_landmark", "addr_missing"]


def context_features(scored: pl.DataFrame) -> pl.DataFrame:
    """scored: s1_rid, src, cand_rid, score (the cheap rule score from match_v2).
    Adds rank/gap features within each S1 and within each candidate (reverse view)."""
    return scored.with_columns(
        # among this S1's candidates
        pl.col("score").rank("ordinal", descending=True).over("s1_rid").alias("rank_in_s1"),
        (pl.col("score").max().over("s1_rid") - pl.col("score")).alias("gap_to_best_in_s1"),
        pl.len().over("s1_rid").alias("n_cands_s1"),
        pl.col("score").rank("ordinal", descending=True).over("s1_rid", "src").alias("rank_in_s1_src"),
        # among the S1s that claim this candidate (reverse view)
        pl.col("score").rank("ordinal", descending=True).over("src", "cand_rid").alias("rank_in_cand"),
        (pl.col("score").max().over("src", "cand_rid") - pl.col("score")).alias("gap_to_best_in_cand"),
        pl.len().over("src", "cand_rid").alias("n_s1_for_cand"),
    )


def _num_feats(n1: pl.Series, n2: pl.Series) -> dict:
    a, b = n1.to_list(), n2.to_list()
    first_eq, jac, prefix, any_missing = [], [], [], []
    for x, y in zip(a, b):
        x = x or []
        y = y or []
        if not x or not y:
            first_eq.append(np.nan); jac.append(np.nan); prefix.append(np.nan); any_missing.append(1.0)
            continue
        sx, sy = set(x), set(y)
        first_eq.append(float(x[0] == y[0]))
        jac.append(len(sx & sy) / len(sx | sy))
        prefix.append(float(any(u != v and (u.startswith(v) or v.startswith(u)) for u in sx for v in sy)))
        any_missing.append(0.0)
    return {"num_first_equal": first_eq, "num_jaccard": jac, "num_prefix": prefix, "num_missing": any_missing}


def pair_features(pairs: pl.DataFrame, c1: pl.DataFrame, c23: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_rid, src, cand_rid (+ any context columns). c1 / c23: records with
    FEAT_COLS_RECORD. Returns pairs with ~25 extra float32 feature columns."""
    r1 = c1.select([pl.col(c).alias(f"{c}_1") for c in FEAT_COLS_RECORD if c != "src"])
    r2 = c23.select([pl.col(c).alias(f"{c}_2") if c not in ("rid", "src") else pl.col(c)
                     for c in FEAT_COLS_RECORD])
    p = (pairs.join(r1, left_on="s1_rid", right_on="rid_1")
         .join(r2.rename({"rid": "cand_rid"}), on=["src", "cand_rid"]))

    def sim(col, scorer):
        return cpdist(p[f"{col}_1"].to_list(), p[f"{col}_2"].to_list(), scorer=scorer, workers=-1)

    f = {
        "name_token_set": sim("name_core", fuzz.token_set_ratio),
        "name_token_sort": sim("name_core", fuzz.token_sort_ratio),
        "name_ratio": sim("name_core", fuzz.ratio),
        "name_partial": sim("name_core", fuzz.partial_ratio),
        "skel_token_set": sim("name_skel", fuzz.token_set_ratio),
        "compact_ratio": sim("name_compact", fuzz.ratio),
        "compact_jw": sim("name_compact", JaroWinkler.normalized_similarity),
        "addr_token_set": sim("addr_clean", fuzz.token_set_ratio),
        "addr_token_sort": sim("addr_clean", fuzz.token_sort_ratio),
        "addr_partial": sim("addr_clean", fuzz.partial_ratio),
    }
    f.update(_num_feats(p["addr_numbers_1"], p["addr_numbers_2"]))
    lf1, lf2 = p["legal_form_1"].to_numpy(), p["legal_form_2"].to_numpy()
    both = (lf1 != "") & (lf2 != "")
    f["legal_equal"] = np.where(both, (lf1 == lf2).astype(float), np.nan)
    f["legal_one_missing"] = ((lf1 == "") ^ (lf2 == "")).astype(float)
    f["name_exact"] = (p["name_core_1"] == p["name_core_2"]).to_numpy().astype(float)
    f["skel_exact"] = (p["name_skel_1"] == p["name_skel_2"]).to_numpy().astype(float)
    f["len_name_1"] = p["name_core_1"].str.len_chars().to_numpy().astype(float)
    f["len_name_2"] = p["name_core_2"].str.len_chars().to_numpy().astype(float)
    f["web_2"] = p["is_web_2"].to_numpy().astype(float)
    f["nonlatin_2"] = p["name_nonlatin_2"].to_numpy().astype(float)
    f["addr_missing_2"] = p["addr_missing_2"].to_numpy().astype(float)
    f["landmark_any"] = (p["is_landmark_1"] | p["is_landmark_2"]).to_numpy().astype(float)

    # address missing -> address similarities are meaningless: make them NaN
    miss = f["addr_missing_2"] == 1
    for k in ("addr_token_set", "addr_token_sort", "addr_partial"):
        f[k] = np.where(miss, np.nan, f[k])

    keep = [c for c in pairs.columns]
    out = p.select(keep).with_columns([pl.Series(k, np.asarray(v, dtype=np.float32)) for k, v in f.items()])
    return out


CONTEXT_COLS = ["score", "rank_in_s1", "gap_to_best_in_s1", "n_cands_s1", "rank_in_s1_src",
                "rank_in_cand", "gap_to_best_in_cand", "n_s1_for_cand"]
PAIR_COLS = ["name_token_set", "name_token_sort", "name_ratio", "name_partial", "skel_token_set",
             "compact_ratio", "compact_jw", "addr_token_set", "addr_token_sort", "addr_partial",
             "num_first_equal", "num_jaccard", "num_prefix", "num_missing", "legal_equal",
             "legal_one_missing", "name_exact", "skel_exact", "len_name_1", "len_name_2",
             "web_2", "nonlatin_2", "addr_missing_2", "landmark_any"]
FEATURES = ["src"] + CONTEXT_COLS + PAIR_COLS