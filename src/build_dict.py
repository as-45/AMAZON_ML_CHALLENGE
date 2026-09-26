"""Step 3: learn an Indian-script -> English word dictionary from the TRAINING pairs.

Idea: S2/S3 often write an S1 name in Hindi, Gujarati, Telugu, ... script, word by word:
    S1  "Jain Construction Private Limited"
    S2  "जैन कंस्ट्रक्शन प्राइवेट लिमिटेड"
When both names have the same number of words, word i on one side is the translation of
word i on the other. Counting these pairs over ~500k training matches gives a reliable
dictionary (only labelled TRAIN data is used; the test files are never read here).

    python -m src.build_dict --data ../../student_resource/dataset --work work
Writes work/indic_dict.json, which normalize.py uses automatically on the next prepare run.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from . import config
from .ingest import read_ground_truth_pairs, read_source

NON_LATIN = re.compile(r"[^\x00-\u024F]")
MIN_COUNT = 3        # a mapping must be seen at least this often
MIN_SHARE = 0.6      # and be the translation in >= 60% of its sightings


def words(name: str) -> list:
    return [w for w in re.split(r"\s+", name.strip()) if w]


def latin_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def build(data_dir: Path) -> dict:
    pairs = read_ground_truth_pairs(data_dir)
    s1 = read_source(data_dir, "train", 1).select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"))
    s23 = pl.concat([read_source(data_dir, "train", s) for s in (2, 3)]).select(
        pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("n2"))
    s23 = s23.filter(pl.col("n2").str.contains(r"[^\x00-\u024F]"))
    p = pairs.join(s23, on="cand_id").join(s1, on="s1_id")
    print(f"training pairs with a non-Latin name: {p.height:,}")

    counts = defaultdict(Counter)
    aligned = 0
    for n1, n2 in p.select("n1", "n2").iter_rows():
        a, b = words(n1), words(n2)
        if len(a) != len(b):
            continue
        aligned += 1
        for x, y in zip(a, b):
            if NON_LATIN.search(y):
                lx = latin_word(x)
                if lx:
                    counts[y][lx] += 1
    table = {}
    for native, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE:
            table[native] = best
    print(f"aligned pairs: {aligned:,}  native words seen: {len(counts):,}  kept in dictionary: {len(table):,}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.DEFAULT_DATA_DIR)
    ap.add_argument("--work", type=Path, default=config.DEFAULT_WORK_DIR)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    table = build(args.data)
    out = args.work / "indic_dict.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    sample = list(table.items())[:8]
    print("examples:", ", ".join(f"{k}->{v}" for k, v in sample))
    print(f"saved {out}")


if __name__ == "__main__":
    main()