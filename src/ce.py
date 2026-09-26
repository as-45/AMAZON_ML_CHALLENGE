"""Step 6: character-level cross-encoder (a small Transformer trained from scratch, no downloads).

What it does, in one line: it reads BOTH records together, letter by letter, and outputs how
likely they are the same business. The gradient-boosting model compares records through ~40
hand-made similarity numbers; this model learns its own comparisons directly from the text
(e.g. "the house number changed from 12 to 15", "only the legal suffix differs").
Its score is then given to the LightGBM models as three extra features (ce, ce_gap, ce_rank).

Input text per record:  "<core name> | <legal form> | <clean address>"   (lower-case ASCII)
Model input:            [CLS] record A (100 chars) [SEP] record B (100 chars)
Model:                  4-layer Transformer encoder, width 256, 8 heads (~3.4M parameters)

No leakage (important, otherwise LightGBM would over-trust the score):
  train S1 are split into 2 folds by a fixed hash. Model A learns from fold-0 pairs and scores
  fold-1 pairs; model B learns from fold 1 and scores fold 0. Held-out S1 (the ones our eval
  measures on) are never used for learning. Test pairs get the average of model A and model B.

Stages (see model_v4.py):
  --stage dump   (CPU)  pruned candidate pairs + their texts -> work/v4/ce/pairs_{split}_{country}.parquet
  --stage ce     (GPU)  train the two models, score all pairs -> work/v4/ce/ce_{split}_{country}.parquet
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

A_LEN = B_LEN = 100
SEQ = 1 + A_LEN + 1 + B_LEN          # [CLS] A [SEP] B
VOCAB = 131                          # 0 pad, 1 CLS, 2 SEP, 3.. = ASCII byte + 3
TRAIN_PAIRS = int(os.environ.get("CE_TRAIN_PAIRS", 3_000_000))   # pairs per fold model (sampled)
EPOCHS = int(os.environ.get("CE_EPOCHS", 2))
BATCH = 512
LR = 3e-4
SEED = 13


def ce_dir(work: Path) -> Path:
    d = Path(work) / "v4" / "ce"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_text(df: pl.DataFrame) -> pl.Series:
    return pl.concat_str([pl.col("name_core").fill_null(""), pl.col("legal_form").fill_null(""),
                          pl.col("addr_clean").fill_null("")], separator=" | ").str.to_lowercase().alias("text")


def encode(strings, width: int) -> np.ndarray:
    """uint8 [n, width]: ASCII byte + 3, 0 = padding (non-ASCII characters dropped)."""
    out = np.zeros((len(strings), width), dtype=np.uint8)
    for i, s in enumerate(strings):
        b = np.frombuffer(s.encode("ascii", "ignore")[:width], dtype=np.uint8)
        out[i, :len(b)] = np.minimum(b, 127) + 3
    return out


def encode_pairs(ta, tb) -> np.ndarray:
    x = np.zeros((len(ta), SEQ), dtype=np.uint8)
    x[:, 0] = 1
    x[:, 1:1 + A_LEN] = encode(ta, A_LEN)
    x[:, 1 + A_LEN] = 2
    x[:, 2 + A_LEN:] = encode(tb, B_LEN)
    return x


def build_model():
    import torch
    from torch import nn

    class CrossEncoder(nn.Module):
        def __init__(self, d=256, layers=4, heads=8):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, d, padding_idx=0)
            self.pos = nn.Embedding(SEQ, d)
            self.seg = nn.Embedding(2, d)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
            self.register_buffer("positions", torch.arange(SEQ), persistent=False)
            self.register_buffer("segments", (torch.arange(SEQ) > A_LEN + 1).long(), persistent=False)

        def forward(self, x):
            x = x.long()
            h = self.tok(x) + self.pos(self.positions) + self.seg(self.segments)
            pad = x == 0
            h = self.enc(h, src_key_padding_mask=pad)
            return self.head(self.norm(h[:, 0])).squeeze(-1)

    return CrossEncoder()


def train_one(X, y, device, log_every=500):
    """Train a fresh model on uint8 inputs X [n, SEQ] and labels y [n]."""
    import torch
    torch.manual_seed(SEED)
    model = build_model().to(device)
    Xg = torch.from_numpy(X).to(device)
    yg = torch.from_numpy(y.astype(np.float32)).to(device)
    n = len(Xg)
    steps = EPOCHS * math.ceil(n / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = min(1000, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * max(0.05, 1 - s / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    model.train()
    t, step, run = time.time(), 0, 0.0
    g = torch.Generator(device=device).manual_seed(SEED)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=device, generator=g)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = lossf(model(Xg[idx]).float(), yg[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % log_every == 0 or step == steps:
                print(f"      epoch {ep + 1} step {step:,}/{steps:,} loss {run:.4f} ({time.time() - t:.0f}s)", flush=True)
    del Xg, yg
    return model


def predict(model, X, device) -> np.ndarray:
    import torch
    chunk = 4096 if device.type == "cuda" else 256
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            xb = torch.from_numpy(X[s:s + chunk]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out[s:s + chunk] = torch.sigmoid(model(xb).float()).cpu().numpy()
    return out


def stage_ce(work: Path, is_holdout_expr) -> None:
    import torch
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ce] device: {device}", flush=True)
    d = ce_dir(work)
    files = {p.name[len("pairs_"):-len(".parquet")]: p for p in sorted(d.glob("pairs_*.parquet"))}
    train_keys = [k for k in files if k.startswith("train_")]
    test_keys = [k for k in files if k.startswith("test_")]
    tr = pl.concat([pl.read_parquet(files[k]).with_columns(pl.lit(k).alias("part")) for k in train_keys])
    tr = tr.with_columns((pl.col("s1_rid").hash(seed=SEED) % 2).alias("fold"), is_holdout_expr.alias("hold"))
    print(f"[ce] train pairs {tr.height:,} ({tr['label'].mean():.3f} positive) ({time.time() - t0:.0f}s)", flush=True)
    X_all = encode_pairs(tr["ta"].to_list(), tr["tb"].to_list())
    print(f"[ce] encoded train pairs ({time.time() - t0:.0f}s)", flush=True)
    fold = tr["fold"].to_numpy()
    learn = ~tr["hold"].to_numpy()
    y = tr["label"].to_numpy().astype(np.int8)
    score_tr = np.zeros(tr.height, dtype=np.float32)
    models = []
    rng = np.random.default_rng(SEED)
    for f in (0, 1):
        idx = np.flatnonzero((fold == f) & learn)
        if len(idx) > TRAIN_PAIRS:
            idx = np.sort(rng.choice(idx, TRAIN_PAIRS, replace=False))
        print(f"[ce] model {f}: learning from {len(idx):,} pairs of fold {f}", flush=True)
        m = train_one(X_all[idx], y[idx], device)
        other = np.flatnonzero(fold != f)
        score_tr[other] = predict(m, X_all[other], device)
        yo = y[other]
        p = np.clip(score_tr[other], 1e-6, 1 - 1e-6)
        ll = -np.mean(yo * np.log(p) + (1 - yo) * np.log(1 - p))
        acc = np.mean((p > 0.5) == (yo == 1))
        print(f"[ce] model {f}: out-of-fold logloss {ll:.4f}  accuracy {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        models.append(m)
    del X_all
    out = tr.select("part", "s1_rid", "src", "cand_rid").with_columns(pl.Series("ce", score_tr))
    for k in train_keys:
        out.filter(pl.col("part") == k).drop("part").write_parquet(d / f"ce_{k}.parquet")
    del tr, out
    for k in test_keys:
        te = pl.read_parquet(files[k])
        s = np.zeros(te.height, dtype=np.float32)
        for c in range(0, te.height, 2_000_000):
            part = te.slice(c, 2_000_000)
            Xt = encode_pairs(part["ta"].to_list(), part["tb"].to_list())
            s[c:c + part.height] = np.mean([predict(m, Xt, device) for m in models], axis=0)
        te.select("s1_rid", "src", "cand_rid").with_columns(pl.Series("ce", s)).write_parquet(d / f"ce_{k}.parquet")
        print(f"[ce] scored {k}: {te.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    print(f"[ce] done ({time.time() - t0:.0f}s)", flush=True)


def load_ce(work: Path, split: str, country: str):
    p = ce_dir(work) / f"ce_{split}_{country}.parquet"
    return pl.read_parquet(p) if p.exists() else None


def add_ce(feats: pl.DataFrame, ce) -> pl.DataFrame:
    """Join the cross-encoder score; pairs it never saw get null (LightGBM treats as missing)."""
    if ce is None:
        f = feats.with_columns(pl.lit(None, pl.Float32).alias("ce"))
    else:
        f = feats.join(ce, on=["s1_rid", "src", "cand_rid"], how="left")
    return f.with_columns(
        (pl.col("ce") - pl.col("ce").max().over("s1_rid")).alias("ce_gap"),
        pl.col("ce").rank("ordinal", descending=True).over("s1_rid").cast(pl.Float32).alias("ce_rank"),
    )