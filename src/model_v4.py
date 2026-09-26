"""v4 pipeline: keys + TF-IDF search -> pruner -> rich features -> model -> cross-source
second stage -> expected-F0.5 assignment.

Three stages, each saving to work/v4/ so a failed run can resume:
    python -m src.model_v4 --stage cands --split train   # candidates + cheap features (no model)
    python -m src.model_v4 --stage cands --split test
    python -m src.model_v4 --stage dump                  # pruned pairs + texts for the cross-encoder (CPU)
    python -m src.model_v4 --stage ce                    # train + run the cross-encoder (GPU), see ce.py
    python -m src.model_v4 --stage train                 # pruner, main model, stage-2 model, report
    python -m src.model_v4 --stage eval                  # test-like score estimate (all S1 compete)
    python -m src.model_v4 --stage predict               # output/matching_results.tsv + candidate_pairs.tsv

`cands` never imports lightgbm (its OpenMP library can clash with sparse_dot_topn's).
Validation: S1 are split by a fixed hash, so every run uses the same held-out businesses.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import blocking, ce, config, match_v2, retrieve
from .features import FEAT_COLS_RECORD, context_features, pair_features
from .features_v4 import (FEATURES_STAGE2, FEATURES_V4, PRUNE_FEATURES, add_stage2,
                          add_v4_extras, name_counts)

LOAD_COLS = sorted(set(FEAT_COLS_RECORD) | {"country", "entity_id"})
S1_CHUNK = 50_000
PRUNE_K = 12          # candidates kept per S1 after the pruner
PRUNE_MIN = 0.001     # ... and only if pruner probability >= this
DUMP_K, DUMP_MIN = 16, 0.0003   # looser cut for the cross-encoder pairs (a superset of the above)
match_v2.MAX_CANDS = 80   # don't cut the key+search union before the pruner sees it


# ---------------------------------------------------------------- helpers
def v4_dir(work: Path) -> Path:
    d = Path(work) / "v4"
    d.mkdir(parents=True, exist_ok=True)
    return d


ONLY_COUNTRIES = None   # set by --countries (for quick tests)
TEST_LIMIT = 0         # set by --test-limit (for quick tests): only the first N test S1 per country


def countries(work: Path, split: str) -> list:
    cs = match_v2.countries(work, split)
    return [c for c in cs if ONLY_COUNTRIES is None or c in ONLY_COUNTRIES]


def load_universe(work: Path, split: str, country: str, universe: int, seed: int = config.SEED):
    """One country's records. For train, S1 is capped at `universe` and S2/S3 cut to the
    same share (true owners + same fraction of decoys), so competition resembles test."""
    scan = lambda s: pl.scan_parquet(config.clean_path(work, split, s)).select(LOAD_COLS).filter(
        pl.col("country") == country).collect()
    c1, c23 = scan(1), pl.concat([scan(2), scan(3)])
    if split == "test" and TEST_LIMIT:
        c1 = c1.head(TEST_LIMIT)
    if split == "train" and c1.height > universe:
        frac = universe / c1.height
        c1 = c1.sample(universe, seed=seed)
        pairs = pl.read_parquet(config.pairs_path(work))
        owned = pairs.join(c1.select(pl.col("rid").alias("s1_rid")), on="s1_rid").select("src", pl.col("cand_rid").alias("rid"))
        decoys = c23.join(pairs.select("src", pl.col("cand_rid").alias("rid")), on=["src", "rid"], how="anti")
        c23 = pl.concat([c23.join(owned, on=["src", "rid"], how="semi"), decoys.sample(fraction=frac, seed=seed)])
    return c1, c23


def s1_group(col: str = "s1_rid") -> pl.Expr:
    """Fixed split of S1 by hash: 0-1 pruner set (20%), 2-9 main set; main set: holdout = 1 in 5."""
    return (pl.col(col).hash(seed=11) % 10).alias("grp")


def is_holdout(col: str = "s1_rid") -> pl.Expr:
    return (pl.col(col).hash(seed=1) % 5 == 0)


# ---------------------------------------------------------------- stage: cands
def country_candidates(c1: pl.DataFrame, c23: pl.DataFrame, threads: int) -> pl.DataFrame:
    t = time.time()
    dfs = blocking.word_df(c23)
    c23k, c1k = blocking.add_keys(c23, dfs), blocking.add_keys(c1, dfs)
    del dfs
    keys = pl.concat([blocking.candidates(c1k.slice(i, S1_CHUNK), c23k) for i in range(0, c1k.height, S1_CHUNK)]).unique()
    del c23k, c1k
    gc.collect()
    print(f"    keys: {keys.height:,} pairs ({time.time()-t:.0f}s)", flush=True)
    t = time.time()
    found = retrieve.search(c1, c23, threads=threads, extra=keys)
    del keys
    gc.collect()
    print(f"    keys + search: {found.height:,} pairs ({time.time()-t:.0f}s)", flush=True)
    t = time.time()
    small = c23.select("rid", "src", "name_core", "name_skel", "addr_clean")
    scored = []
    for i in range(0, c1.height, S1_CHUNK):
        part = c1.slice(i, S1_CHUNK)
        pairs = found.join(part.select(pl.col("rid").alias("s1_rid")), on="s1_rid", how="semi")
        scored.append(match_v2.score(pairs.select("s1_rid", "src", "cand_rid"), part, small))
    scored = pl.concat(scored).join(found, on=["s1_rid", "src", "cand_rid"], how="left")
    scored = context_features(scored).with_columns(
        pl.col("cos_comb").rank("ordinal", descending=True).over("s1_rid").alias("rank_comb_in_s1"))
    print(f"    cheap score + context: {scored.height:,} pairs ({time.time()-t:.0f}s)", flush=True)
    return scored


def stage_cands(work: Path, split: str, universe: int, threads: int) -> None:
    for country in countries(work, split):
        t = time.time()
        print(f"[cands] {split} {country}", flush=True)
        c1, c23 = load_universe(work, split, country, universe)
        out = country_candidates(c1, c23, threads)
        path = v4_dir(work) / f"cands_{split}_{country}.parquet"
        out.write_parquet(path)
        truth = pl.read_parquet(config.pairs_path(work)) if split == "train" else None
        msg = ""
        if truth is not None:
            tp = truth.join(c1.select(pl.col("rid").alias("s1_rid")), on="s1_rid")
            msg = f" | recall {out.join(tp, on=['s1_rid', 'src', 'cand_rid']).height / tp.height:.2%}"
        print(f"  {country}: S1={c1.height:,} S23={c23.height:,} pairs={out.height:,} "
              f"({out.height / c1.height:.0f}/S1){msg} -> {path.name} ({time.time()-t:.0f}s)", flush=True)
        del c1, c23, out
        gc.collect()


# ---------------------------------------------------------------- shared model steps
def prune(lgb_pruner, cands: pl.DataFrame) -> pl.DataFrame:
    p0 = lgb_pruner.predict(cands.select(PRUNE_FEATURES).to_numpy().astype(np.float32))
    c = cands.with_columns(pl.Series("p0", p0.astype(np.float32)))
    return (c.filter(pl.col("p0") >= PRUNE_MIN)
            .sort("p0", descending=True).group_by("s1_rid", maintain_order=True).head(PRUNE_K))


def rich(pruned: pl.DataFrame, c1: pl.DataFrame, c23: pl.DataFrame, counts, ce_scores=None) -> pl.DataFrame:
    out = []
    ids = pruned["s1_rid"].unique()
    for i in range(0, len(ids), S1_CHUNK):
        chunk_ids = ids.slice(i, S1_CHUNK).implode()
        part = pruned.filter(pl.col("s1_rid").is_in(chunk_ids))
        f = pair_features(part, c1.filter(pl.col("rid").is_in(chunk_ids)), c23)
        out.append(add_v4_extras(f, c1, c23, counts))
    return ce.add_ce(pl.concat(out), ce_scores)


def X(df: pl.DataFrame, cols) -> np.ndarray:
    return df.select(cols).to_numpy().astype(np.float32)


def assign_expected(pred: pl.DataFrame) -> pl.DataFrame:
    from .model import assign_expected as _ae
    return _ae(pred)


def f05(matched: pl.DataFrame, truth_pairs: pl.DataFrame, s1_rids) -> float:
    from .model import f05_of
    return f05_of(matched, truth_pairs, s1_rids)


# ---------------------------------------------------------------- stage: dump (for the cross-encoder)
def stage_dump(work: Path, universe: int) -> None:
    """Pruned candidate pairs (looser cut than PRUNE_K/PRUNE_MIN) with both records' texts."""
    import lightgbm as lgb
    t0 = time.time()
    d = v4_dir(work)
    pruner = lgb.Booster(model_file=str(d / "pruner.lgb"))
    truth = pl.read_parquet(config.pairs_path(work)).with_columns(pl.lit(1, pl.Int8).alias("label"))
    for split in ("train", "test"):
        for country in countries(work, split):
            path = d / f"cands_{split}_{country}.parquet"
            if not path.exists():
                print(f"  {split} {country}: no {path.name}, skipped", flush=True)
                continue
            cands = pl.read_parquet(path)
            p0 = pruner.predict(cands.select(PRUNE_FEATURES).to_numpy().astype(np.float32))
            pairs = (cands.select("s1_rid", "src", "cand_rid").with_columns(pl.Series("p0", p0.astype(np.float32)))
                     .filter(pl.col("p0") >= DUMP_MIN).sort("p0", descending=True)
                     .group_by("s1_rid", maintain_order=True).head(DUMP_K).drop("p0"))
            del cands
            c1, c23 = load_universe(work, split, country, universe)
            t1 = c1.select(pl.col("rid").alias("s1_rid"), ce.record_text(c1).alias("ta"))
            t2 = c23.select("src", pl.col("rid").alias("cand_rid"), ce.record_text(c23).alias("tb"))
            pairs = pairs.join(t1, on="s1_rid", how="left").join(t2, on=["src", "cand_rid"], how="left")
            if split == "train":
                pairs = pairs.join(truth, on=["s1_rid", "src", "cand_rid"], how="left").with_columns(
                    pl.col("label").fill_null(0))
            pairs.write_parquet(ce.ce_dir(work) / f"pairs_{split}_{country}.parquet")
            print(f"  {split} {country}: {pairs.height:,} pairs ({pairs.height / c1.height:.1f}/S1) "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del pairs, c1, c23, t1, t2
            gc.collect()


# ---------------------------------------------------------------- stage: train
def stage_train(work: Path, universe: int, sample: int, refit_pruner: bool = False) -> None:
    import lightgbm as lgb
    t0 = time.time()
    d = v4_dir(work)
    truth = pl.read_parquet(config.pairs_path(work))
    lab = truth.with_columns(pl.lit(1, pl.Int8).alias("label"))
    P = dict(objective="binary", verbose=-1, seed=config.SEED, num_threads=0)

    # 1) pruner, trained on group 0-1 S1 of every country
    per_country, prune_rows = {}, []
    for country in countries(work, "train"):
        cands = pl.read_parquet(d / f"cands_train_{country}.parquet").with_columns(s1_group())
        s1_ids = cands["s1_rid"].unique().sample(min(sample, cands["s1_rid"].n_unique()), seed=config.SEED)
        cands = cands.join(s1_ids.to_frame("s1_rid"), on="s1_rid", how="semi")
        cands = cands.join(lab, on=["s1_rid", "src", "cand_rid"], how="left").with_columns(pl.col("label").fill_null(0))
        prune_rows.append(cands.filter(pl.col("grp") < 2))
        per_country[country] = cands.filter(pl.col("grp") >= 2).drop("grp")
    if (d / "pruner.lgb").exists() and not refit_pruner:
        # keep the pruner the cross-encoder pairs were cut with (stage dump), so every pair has a score
        pruner = lgb.Booster(model_file=str(d / "pruner.lgb"))
        print("[train] using the saved pruner (pass --refit-pruner to retrain it)", flush=True)
    else:
        pr = pl.concat(prune_rows)
        pruner = lgb.train(dict(P, learning_rate=0.1, num_leaves=63, min_data_in_leaf=100),
                           lgb.Dataset(X(pr, PRUNE_FEATURES), pr["label"].to_numpy()), num_boost_round=300)
        pruner.save_model(str(d / "pruner.lgb"))
        print(f"[train] pruner fitted on {pr.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
        del pr
    del prune_rows

    # 2) prune + rich features for the main set
    rows, recs = [], {}
    for country, cands in per_country.items():
        t = time.time()
        pruned = prune(pruner, cands)
        c1, c23 = load_universe(work, "train", country, universe)
        feats = rich(pruned, c1, c23, name_counts(c1, c23), ce.load_ce(work, "train", country)).with_columns(
            pl.lit(country).alias("country"))
        tp = truth.join(cands.select("s1_rid").unique(), on="s1_rid")
        rec_before = cands["label"].sum() / tp.height
        rec_after = feats["label"].sum() / tp.height
        print(f"  {country}: recall before pruning {rec_before:.2%} -> after {rec_after:.2%} "
              f"({feats.height / cands['s1_rid'].n_unique():.1f} cands/S1) ({time.time()-t:.0f}s)", flush=True)
        rows.append(feats)
        recs[country] = c23.select("src", "rid", "name_core", "addr_clean")
        del c1, c23, cands, pruned
        gc.collect()
    data = pl.concat(rows).with_columns(is_holdout().alias("hold"), (pl.col("s1_rid").hash(seed=5) % 2).alias("fold"))
    del rows
    tr, ho = data.filter(~pl.col("hold")), data.filter(pl.col("hold"))
    inner = tr["s1_rid"].hash(seed=9) % 10 == 0

    # 3) main model (stage 1) + out-of-fold p1 on the training part
    P1 = dict(P, learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1)

    def fit(df, es):
        m = lgb.train(P1, lgb.Dataset(X(df, FEATURES_V4), df["label"].to_numpy()), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X(es, FEATURES_V4), es["label"].to_numpy())],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return m
    trn, es = tr.filter(~inner), tr.filter(inner)
    m1 = fit(trn, es)
    oof = np.zeros(tr.height, dtype=np.float32)
    for k in (0, 1):
        mk = fit(trn.filter(pl.col("fold") != k), es)
        idx = np.flatnonzero((tr["fold"] == k).to_numpy())
        oof[idx] = mk.predict(X(tr[idx], FEATURES_V4), num_iteration=mk.best_iteration)
    tr = tr.with_columns(pl.Series("p1", oof))
    ho = ho.with_columns(pl.Series("p1", m1.predict(X(ho, FEATURES_V4), num_iteration=m1.best_iteration).astype(np.float32)))
    m1.save_model(str(d / "stage1.lgb"), num_iteration=m1.best_iteration)
    print(f"[train] stage 1: {m1.best_iteration} rounds ({time.time()-t0:.0f}s)", flush=True)

    # 4) stage 2: cross-source agreement features, trained on out-of-fold p1
    def s2(df):
        return pl.concat([add_stage2(df.filter(pl.col("country") == c), recs[c]) for c in recs])
    tr2, ho2 = s2(tr), s2(ho)
    inner2 = tr2["s1_rid"].hash(seed=9) % 10 == 0
    m2 = lgb.train(P1, lgb.Dataset(X(tr2.filter(~inner2), FEATURES_STAGE2), tr2.filter(~inner2)["label"].to_numpy()),
                   num_boost_round=3000,
                   valid_sets=[lgb.Dataset(X(tr2.filter(inner2), FEATURES_STAGE2), tr2.filter(inner2)["label"].to_numpy())],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    ho2 = ho2.with_columns(pl.Series("p2", m2.predict(X(ho2, FEATURES_STAGE2), num_iteration=m2.best_iteration).astype(np.float32)))
    m2.save_model(str(d / "stage2.lgb"), num_iteration=m2.best_iteration)
    print(f"[train] stage 2: {m2.best_iteration} rounds ({time.time()-t0:.0f}s)", flush=True)

    # 5) report on the held-out S1 (true pairs missed by candidates count as misses)
    hold_s1 = ho2["s1_rid"].unique().to_list()
    tp_hold = truth.join(ho2.select("s1_rid").unique(), on="s1_rid")
    res = {}
    for name, col in (("stage1", "p1"), ("stage2", "p2")):
        pred = ho2.select("s1_rid", "src", "cand_rid", pl.col(col).alias("p"))
        res[name] = f05(assign_expected(pred), tp_hold, hold_s1)
        by_c = {}
        for c in sorted(recs):
            hc = ho2.filter(pl.col("country") == c)
            by_c[c] = f05(assign_expected(hc.select("s1_rid", "src", "cand_rid", pl.col(col).alias("p"))),
                          truth.join(hc.select("s1_rid").unique(), on="s1_rid"), hc["s1_rid"].unique().to_list())
        print(f"  HOLDOUT {name}: macro F0.5 = {res[name]:.4f}  " + "  ".join(f"{c} {v:.4f}" for c, v in by_c.items()), flush=True)
    # unseen-country check (proxy for France): train stage 1 on one country, test on the other
    for a_c in sorted(recs):
        for b_c in sorted(recs):
            if a_c == b_c:
                continue
            ta = trn.filter(pl.col("country") == a_c)
            ea = es.filter(pl.col("country") == a_c)
            mx = fit(ta, ea)
            hb = ho.filter(pl.col("country") == b_c)
            pb = hb.select("s1_rid", "src", "cand_rid").with_columns(
                pl.Series("p", mx.predict(X(hb, FEATURES_V4), num_iteration=mx.best_iteration).astype(np.float32)))
            fb = f05(assign_expected(pb), truth.join(hb.select("s1_rid").unique(), on="s1_rid"), hb["s1_rid"].unique().to_list())
            print(f"  UNSEEN-COUNTRY check: train on {a_c} only -> {b_c} holdout F0.5 = {fb:.4f}", flush=True)
    ceiling = f05(ho2.filter(pl.col("label") == 1).select("s1_rid", "src", "cand_rid"), tp_hold, hold_s1)
    print(f"  ceiling (perfect model on these candidates): {ceiling:.4f}", flush=True)
    use = "stage2" if res["stage2"] > res["stage1"] else "stage1"
    (d / "config.json").write_text(json.dumps({"use": use, "holdout": res, "ceiling": ceiling}, indent=2))
    print(f"[train] done: using {use} ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- stage: eval
def stage_eval(work: Path, universe: int) -> None:
    """Test-like score estimate, no upload needed.
    The train report scores only a small holdout, so few S1 compete for the same S2/S3 records.
    Here we run the saved models on ALL train-universe S1 of each country (like predict does on
    test), apply one-owner + expected-F0.5 to everyone at once, and then measure F0.5 only on
    S1 that no model ever trained on (holdout AND not in the pruner group)."""
    import lightgbm as lgb
    t0 = time.time()
    d = v4_dir(work)
    pruner = lgb.Booster(model_file=str(d / "pruner.lgb"))
    m1 = lgb.Booster(model_file=str(d / "stage1.lgb"))
    m2 = lgb.Booster(model_file=str(d / "stage2.lgb"))
    truth = pl.read_parquet(config.pairs_path(work))
    preds, clean_ids = [], []
    for country in countries(work, "train"):
        t = time.time()
        cands = pl.read_parquet(d / f"cands_train_{country}.parquet")
        c1, c23 = load_universe(work, "train", country, universe)
        pruned = prune(pruner, cands)
        feats = rich(pruned, c1, c23, name_counts(c1, c23), ce.load_ce(work, "train", country))
        feats = feats.with_columns(pl.Series("p1", m1.predict(X(feats, FEATURES_V4)).astype(np.float32)))
        feats = add_stage2(feats, c23.select("src", "rid", "name_core", "addr_clean"))
        feats = feats.with_columns(pl.Series("p2", m2.predict(X(feats, FEATURES_STAGE2)).astype(np.float32)),
                                   pl.lit(country).alias("country"))
        preds.append(feats.select("s1_rid", "src", "cand_rid", "p1", "p2", "country"))
        # never-trained S1: holdout (excluded from stage 1/2) and group >= 2 (excluded from pruner)
        ids = c1.select(pl.col("rid").alias("s1_rid"), pl.lit(country).alias("country")).filter(
            is_holdout() & (pl.col("s1_rid").hash(seed=11) % 10 >= 2))
        clean_ids.append(ids)
        print(f"  {country}: {c1.height:,} S1 scored, {ids.height:,} never-trained S1 for the estimate "
              f"({time.time()-t:.0f}s)", flush=True)
        del c1, c23, cands, pruned, feats
        gc.collect()
    pred, clean = pl.concat(preds), pl.concat(clean_ids)
    res = {}
    for name, col in (("stage1", "p1"), ("stage2", "p2")):
        # one-owner + expected-F0.5 over ALL S1 of the country together (as on test)
        matched = assign_expected(pred.select("s1_rid", "src", "cand_rid", pl.col(col).alias("p")))
        parts = []
        for c in sorted(clean["country"].unique()):
            ids = clean.filter(pl.col("country") == c)["s1_rid"]
            m = matched.join(ids.to_frame(), on="s1_rid", how="semi")
            parts.append(f"{c} {f05(m, truth.join(ids.to_frame(), on='s1_rid'), ids.to_list()):.4f}")
        ids = clean["s1_rid"]
        m = matched.join(ids.to_frame(), on="s1_rid", how="semi")
        res[name] = f05(m, truth.join(ids.to_frame(), on="s1_rid"), ids.to_list())
        print(f"  EVAL (all S1 compete, scored on never-trained S1) {name}: macro F0.5 = {res[name]:.4f}  "
              + "  ".join(parts), flush=True)
    (d / "eval.json").write_text(json.dumps(res, indent=2))
    print(f"[eval] done ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- stage: predict
def stage_predict(work: Path, out_dir: Path) -> None:
    import lightgbm as lgb
    t0 = time.time()
    d = v4_dir(work)
    cfg = json.loads((d / "config.json").read_text())
    pruner = lgb.Booster(model_file=str(d / "pruner.lgb"))
    m1 = lgb.Booster(model_file=str(d / "stage1.lgb"))
    m2 = lgb.Booster(model_file=str(d / "stage2.lgb"))
    preds = []
    for country in countries(work, "test"):
        t = time.time()
        cands = pl.read_parquet(d / f"cands_test_{country}.parquet")
        pruned = prune(pruner, cands)
        c1, c23 = load_universe(work, "test", country, 0)
        feats = rich(pruned, c1, c23, name_counts(c1, c23), ce.load_ce(work, "test", country))
        feats = feats.with_columns(pl.Series("p1", m1.predict(X(feats, FEATURES_V4)).astype(np.float32)))
        if cfg["use"] == "stage2":
            feats = add_stage2(feats, c23.select("src", "rid", "name_core", "addr_clean"))
            feats = feats.with_columns(pl.Series("p", m2.predict(X(feats, FEATURES_STAGE2)).astype(np.float32)))
        else:
            feats = feats.with_columns(pl.col("p1").alias("p"))
        preds.append(feats.select("s1_rid", "src", "cand_rid", "p"))
        print(f"  {country}: {c1.height:,} S1, {feats.height:,} pairs scored ({time.time()-t:.0f}s)", flush=True)
        del c1, c23, cands, pruned, feats
        gc.collect()
    pred = pl.concat(preds)
    matched = assign_expected(pred)
    ids = ["rid", "src", "entity_id"]
    s1 = pl.read_parquet(config.clean_path(work, "test", 1), columns=ids)
    s23 = pl.concat([pl.read_parquet(config.clean_path(work, "test", s), columns=ids) for s in (2, 3)])
    from .baseline import to_lists
    out_dir.mkdir(parents=True, exist_ok=True)
    to_lists(pred, s1, s23, "candidate_entity_ids").write_csv(out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    to_lists(matched, s1, s23, "matched_entity_ids").write_csv(out_dir / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"test S1={s1.height:,} candidates={pred.height:,} matches={matched.height:,} "
          f"-> {out_dir} ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cands", "dump", "ce", "train", "eval", "predict"], required=True)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--work", type=Path, default=config.DEFAULT_WORK_DIR)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--universe", type=int, default=800_000, help="max train S1 per country")
    ap.add_argument("--sample", type=int, default=250_000, help="train S1 per country used for models")
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--countries", nargs="*", default=None, help="only these countries (quick tests)")
    ap.add_argument("--test-limit", type=int, default=0, help="only first N test S1 per country (quick tests)")
    ap.add_argument("--refit-pruner", action="store_true", help="retrain the pruner instead of reusing it")
    a = ap.parse_args()
    global ONLY_COUNTRIES, TEST_LIMIT
    ONLY_COUNTRIES, TEST_LIMIT = a.countries, a.test_limit
    if a.stage == "cands":
        stage_cands(a.work, a.split, a.universe, a.threads)
    elif a.stage == "dump":
        stage_dump(a.work, a.universe)
    elif a.stage == "ce":
        ce.stage_ce(a.work, is_holdout())
    elif a.stage == "train":
        stage_train(a.work, a.universe, a.sample, a.refit_pruner)
    elif a.stage == "eval":
        stage_eval(a.work, a.universe)
    else:
        stage_predict(a.work, a.out)


if __name__ == "__main__":
    main()