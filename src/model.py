"""Stages 3-5 with a trained model (uploads #3 and later).

    python -m src.model --mode train      # build training pairs, train LightGBM, pick threshold
    python -m src.model --mode predict    # score test pairs with the saved model, write output/

How it works, per country (the country list comes from the data, so France is included):
  1. blocking (blocking.py) + cheap rule score (match_v2.score) for every S1 -> candidates
  2. context/reverse features over ALL candidates of the country (features.context_features)
  3. pair features for the S1s we need (features.pair_features)
  4. train: LightGBM learns P(match) from labelled train pairs
     predict: model gives P(match) for every test pair
  5. one owner per S2/S3 record, keep pairs with P >= threshold (threshold tuned for macro F0.5)
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import blocking, config
from .baseline import to_lists
from .evaluate import macro_f05
from .features import FEAT_COLS_RECORD, FEATURES, context_features, pair_features
from .match_v2 import countries, score

LOAD_COLS = sorted(set(FEAT_COLS_RECORD) | {"country", "entity_id"})
S1_CHUNK = 50_000          # S1 rows blocked/featurised at a time (memory)
TRAIN_UNIVERSE = 800_000   # max S1 per country used to build train candidates (≈ test size)
TRAIN_SAMPLE = 120_000     # S1 per country whose pairs become training rows
MODEL_PATH = "model.lgb"
CONFIG_PATH = "model_config.json"


# ---------------------------------------------------------------- data building
def load_universe(work: Path, split: str, country: str, seed: int = config.SEED):
    """One country's S1 and S2/S3 records. For train, S1 is capped at TRAIN_UNIVERSE and
    S2/S3 is cut to the same fraction (their true owners + the same share of decoys),
    so candidate competition looks like the (smaller) test set."""
    scan = lambda s: pl.scan_parquet(config.clean_path(work, split, s)).select(LOAD_COLS).filter(
        pl.col("country") == country).collect()
    c1 = scan(1)
    c23 = pl.concat([scan(2), scan(3)])
    if split == "train" and c1.height > TRAIN_UNIVERSE:
        frac = TRAIN_UNIVERSE / c1.height
        c1 = c1.sample(TRAIN_UNIVERSE, seed=seed)
        pairs = pl.read_parquet(config.pairs_path(work))
        owned = pairs.join(c1.select(pl.col("rid").alias("s1_rid")), on="s1_rid").select("src", pl.col("cand_rid").alias("rid"))
        all_owned = pairs.select("src", pl.col("cand_rid").alias("rid"))
        decoys = c23.join(all_owned, on=["src", "rid"], how="anti")
        c23 = pl.concat([
            c23.join(owned, on=["src", "rid"], how="semi"),
            decoys.sample(fraction=frac, seed=seed),
        ])
    return c1, c23


def candidates_with_context(c1: pl.DataFrame, c23: pl.DataFrame) -> pl.DataFrame:
    """Blocking + cheap score for every S1, then context/reverse features."""
    dfs = blocking.word_df(c23)
    c23k = blocking.add_keys(c23, dfs)
    c1k = blocking.add_keys(c1, dfs)
    del dfs
    small = c23.select("rid", "src", "name_core", "name_skel", "addr_clean")
    out = []
    for i in range(0, c1.height, S1_CHUNK):
        part = c1.slice(i, S1_CHUNK)
        cand = blocking.candidates(c1k.join(part.select("rid"), on="rid", how="semi"), c23k)
        out.append(score(cand, part, small))
    return context_features(pl.concat(out))


def featurize(scored: pl.DataFrame, c1: pl.DataFrame, c23: pl.DataFrame, s1_rids: pl.Series):
    """Yield pair-feature chunks for the given S1 rids."""
    for i in range(0, len(s1_rids), S1_CHUNK):
        ids = s1_rids.slice(i, S1_CHUNK)
        chunk = scored.filter(pl.col("s1_rid").is_in(ids.implode()))
        yield pair_features(chunk, c1.filter(pl.col("rid").is_in(ids.implode())), c23)


# ---------------------------------------------------------------- assignment + scoring
def assign_prob(pred: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """One owner per S2/S3 record (highest P), then keep P >= threshold."""
    best = pred.sort("p", "s1_rid", descending=[True, False]).unique(subset=["src", "cand_rid"], keep="first")
    return best.filter(pl.col("p") >= threshold)


def f05_for(pred: pl.DataFrame, truth_pairs: pl.DataFrame, s1_rids, threshold: float) -> float:
    truth = {r: set() for r in s1_rids}
    for a, s, c in truth_pairs.iter_rows():
        truth[a].add((s, c))
    got = {}
    for a, s, c in assign_prob(pred, threshold).select("s1_rid", "src", "cand_rid").iter_rows():
        got.setdefault(a, set()).add((s, c))
    return macro_f05(truth, got)


# ---------------------------------------------------------------- train
def train(work: Path) -> None:
    t0 = time.time()
    truth_all = pl.read_parquet(config.pairs_path(work))
    rows, holdout_s1, blocking_hits, blocking_total = [], [], 0, 0
    for country in countries(work, "train"):
        t = time.time()
        c1, c23 = load_universe(work, "train", country)
        scored = candidates_with_context(c1, c23)
        sample = c1["rid"].sample(min(TRAIN_SAMPLE, c1.height), seed=config.SEED)
        feats = pl.concat(list(featurize(scored, c1, c23, sample)))
        tp = truth_all.join(sample.to_frame("s1_rid"), on="s1_rid")
        feats = feats.join(tp.with_columns(pl.lit(1, pl.Int8).alias("label")),
                           on=["s1_rid", "src", "cand_rid"], how="left").with_columns(pl.col("label").fill_null(0))
        blocking_hits += int(feats["label"].sum())
        blocking_total += tp.height
        rows.append(feats.with_columns(pl.lit(country).alias("country")))
        print(f"  {country}: universe S1={c1.height:,} S23={c23.height:,} | train pairs={feats.height:,} "
              f"({time.time()-t:.0f}s)")
        del c1, c23, scored
    data = pl.concat(rows)
    print(f"blocking recall on sample = {blocking_hits/blocking_total:.1%}  rows={data.height:,}")

    # 80/20 split by S1 (all pairs of one S1 stay together)
    is_hold = (data["s1_rid"].hash(seed=1) % 5 == 0)
    tr, ho = data.filter(~is_hold), data.filter(is_hold)
    X = lambda d: d.select(FEATURES).to_numpy().astype(np.float32)
    dtr = lgb.Dataset(X(tr), tr["label"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
    dho = lgb.Dataset(X(ho), ho["label"].to_numpy(), reference=dtr)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=config.SEED,
                  verbose=-1, num_threads=0)
    booster = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dho],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)])
    print(f"trained {booster.best_iteration} rounds ({time.time()-t0:.0f}s so far)")

    # threshold search on the holdout, with the one-owner rule, per country too
    ho = ho.with_columns(pl.Series("p", booster.predict(X(ho), num_iteration=booster.best_iteration)))
    hold_s1 = ho["s1_rid"].unique().to_list()
    tp_hold = truth_all.join(ho.select("s1_rid").unique(), on="s1_rid")
    best_t, best_f = 0.5, -1.0
    for th in np.arange(0.20, 0.91, 0.05):
        f = f05_for(ho.select("s1_rid", "src", "cand_rid", "p"), tp_hold, hold_s1, th)
        print(f"  threshold {th:.2f}: holdout macro F0.5 = {f:.4f}")
        if f > best_f:
            best_t, best_f = float(th), f
    for country in sorted(ho["country"].unique().to_list()):
        hc = ho.filter(pl.col("country") == country)
        s1c = hc["s1_rid"].unique().to_list()
        f = f05_for(hc.select("s1_rid", "src", "cand_rid", "p"),
                    truth_all.join(hc.select("s1_rid").unique(), on="s1_rid"), s1c, best_t)
        print(f"  {country}: holdout macro F0.5 = {f:.4f} at threshold {best_t:.2f}")

    booster.save_model(str(work / MODEL_PATH), num_iteration=booster.best_iteration)
    (work / CONFIG_PATH).write_text(json.dumps({"threshold": best_t, "holdout_f05": best_f,
                                                "features": FEATURES}, indent=2))
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])[:10]
    print("top features:", ", ".join(k for k, _ in imp))
    print(f"BEST threshold {best_t:.2f}  holdout macro F0.5 = {best_f:.4f}  -> saved {MODEL_PATH} "
          f"({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------- predict
def predict(work: Path, out_dir: Path, threshold=None) -> None:
    t0 = time.time()
    booster = lgb.Booster(model_file=str(work / MODEL_PATH))
    cfg = json.loads((work / CONFIG_PATH).read_text())
    threshold = cfg["threshold"] if threshold is None else threshold
    preds = []
    for country in countries(work, "test"):
        t = time.time()
        c1, c23 = load_universe(work, "test", country)
        scored = candidates_with_context(c1, c23)
        for feats in featurize(scored, c1, c23, c1["rid"]):
            p = booster.predict(feats.select(FEATURES).to_numpy().astype(np.float32))
            preds.append(feats.select("s1_rid", "src", "cand_rid").with_columns(pl.Series("p", p.astype(np.float32))))
        print(f"  {country}: {c1.height:,} S1 scored by model ({time.time()-t:.0f}s)")
        del c1, c23, scored
    pred = pl.concat(preds)
    matched = assign_prob(pred, threshold)
    ids = ["rid", "src", "entity_id"]
    s1 = pl.read_parquet(config.clean_path(work, "test", 1), columns=ids)
    s23 = pl.concat([pl.read_parquet(config.clean_path(work, "test", s), columns=ids) for s in (2, 3)])
    out_dir.mkdir(parents=True, exist_ok=True)
    to_lists(pred, s1, s23, "candidate_entity_ids").write_csv(
        out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    to_lists(matched, s1, s23, "matched_entity_ids").write_csv(
        out_dir / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"test S1={s1.height:,} candidates={pred.height:,} matches={matched.height:,} "
          f"threshold={threshold:.2f} -> {out_dir} ({time.time()-t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict"], required=True)
    ap.add_argument("--work", type=Path, default=config.DEFAULT_WORK_DIR)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    args = ap.parse_args()
    if args.mode == "train":
        train(args.work)
    else:
        predict(args.work, args.out, args.threshold)


def assign_expected(pred: pl.DataFrame) -> pl.DataFrame:
    """Step 1: expected-F0.5 set selection (no fixed threshold).

    1. One owner per S2/S3 record (its highest-P S1); other pairs of that record are dropped.
    2. For each S1, sort its remaining candidates by P (p1 >= p2 >= ...). Keeping the top j gives
         expected F0.5 ~= 1.25 * (p1 + ... + pj) / (0.25 * (sum of all p) + j)
       and keeping NONE gives expected F0.5 = (1 - p1)(1 - p2)... (credit only if truly no match).
    3. Keep whichever of these is highest. Weak candidates everywhere -> empty list automatically.
    """
    owned = pred.sort("p", "s1_rid", descending=[True, False]).unique(subset=["src", "cand_rid"], keep="first")
    ranked = owned.sort(["s1_rid", "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over("s1_rid").alias("cum"),
        pl.col("p").sum().over("s1_rid").alias("tot"),
        pl.int_range(1, pl.len() + 1).over("s1_rid").alias("j"),
        (1 - pl.col("p").clip(0, 1 - 1e-9)).log().sum().over("s1_rid").exp().alias("p_empty"),
    ).with_columns((1.25 * pl.col("cum") / (0.25 * pl.col("tot") + pl.col("j"))).alias("ef"))
    best = ranked.group_by("s1_rid").agg(
        pl.col("j").sort_by("ef", descending=True).first().alias("best_j"),
        pl.col("ef").max().alias("best_ef"), pl.col("p_empty").first())
    keep = best.filter(pl.col("best_ef") > pl.col("p_empty")).select("s1_rid", "best_j")
    return ranked.join(keep, on="s1_rid").filter(pl.col("j") <= pl.col("best_j")).select(pred.columns)


def f05_of(matched: pl.DataFrame, truth_pairs: pl.DataFrame, s1_rids) -> float:
    truth = {r: set() for r in s1_rids}
    for a, s, c in truth_pairs.iter_rows():
        truth[a].add((s, c))
    got = {}
    for a, s, c in matched.select("s1_rid", "src", "cand_rid").iter_rows():
        got.setdefault(a, set()).add((s, c))
    return macro_f05(truth, got)


if __name__ == "__main__":
    main()