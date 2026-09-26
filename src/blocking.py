# This finds candidate matches using 8 keys per record, built from rare words, address numbers and the sound-alike name skeleton.
"""Stage 2 (v2 blocking): find likely matches with rare-word, address and sound-alike keys.

  1. word_df   : count, per country, how many S2/S3 records contain each word
  2. add_keys  : give every record short keys built from its RAREST words, its address
                 numbers and its sound-alike name skeleton
  3. candidates: pair each S1 with the S2/S3 records that share any key
                 (a key shared by more than MAX_BLOCK records is skipped as too common)
"""
import polars as pl

MAX_BLOCK = 50        # skip a key if more than this many S2/S3 records share it
MIN_LEN = 3           # ignore very short words
CHUNK = 300_000       # rows processed at a time (keeps memory low)
KEYS = ("k_name", "k_skel", "k_rn1", "k_rn2", "k_rn1_ra1", "k_skel1_num", "k_num_ra1", "k_ra12")
# (S1 key, S2/S3 key) combinations to join on
KEY_PAIRS = [(k, k) for k in KEYS] + [("k_rn1", "k_rn2"), ("k_rn2", "k_rn1")]


def _words(col: str) -> pl.Expr:
    """Unique words of length >= MIN_LEN that are not pure numbers."""
    return (
        pl.col(col).str.split(" ")
        .list.eval(pl.element().filter((pl.element().str.len_chars() >= MIN_LEN)
                                       & ~pl.element().str.contains(r"^\d+$")))
        .list.unique()
    )


def word_df(s23: pl.DataFrame) -> pl.DataFrame:
    """(country, w, df): in how many S2/S3 records each name/address word appears."""
    total = None
    for i in range(0, s23.height, CHUNK):
        part = s23.slice(i, CHUNK)
        counts = [
            part.select("country", _words(col).alias("w")).explode("w").drop_nulls("w")
            .group_by("country", "w").len()
            for col in ("name_core", "addr_clean")
        ]
        if total is not None:
            counts.append(total)
        total = pl.concat(counts).group_by("country", "w").agg(pl.col("len").sum())
    return total.rename({"len": "df"})


def _rarest(df: pl.DataFrame, col: str, dfs: pl.DataFrame, n: int, prefix: str) -> pl.DataFrame:
    """Add columns prefix1..prefixN = the record's N rarest words in `col`."""
    ex = (
        df.select("rid", "src", "country", _words(col).alias("w")).explode("w").drop_nulls("w")
        .join(dfs, on=["country", "w"], how="left")
        .with_columns(pl.col("df").fill_null(0))
        .sort("df", "w")
        .group_by("src", "rid", maintain_order=True).agg(pl.col("w").head(n))
    )
    cols = [pl.col("w").list.get(i, null_on_oob=True).alias(f"{prefix}{i+1}") for i in range(n)]
    return df.join(ex.select("src", "rid", *cols), on=["src", "rid"], how="left")


def _add_keys(df: pl.DataFrame, dfs: pl.DataFrame) -> pl.DataFrame:
    df = _rarest(df, "name_core", dfs, 2, "rn")
    df = _rarest(df, "addr_clean", dfs, 2, "ra")
    num = pl.col("addr_numbers").list.first()
    skel1 = pl.col("name_skel").str.split(" ").list.first()
    skel = pl.col("name_skel").str.replace_all(" ", "")
    return df.with_columns(
        pl.when(pl.col("name_compact").str.len_chars() >= 4).then(pl.col("name_compact")).alias("k_name"),
        pl.when(skel.str.len_chars() >= 3).then(skel).alias("k_skel"),
        pl.col("rn1").alias("k_rn1"),
        pl.col("rn2").alias("k_rn2"),
        (pl.col("rn1") + "|" + pl.col("ra1")).alias("k_rn1_ra1"),
        pl.when(skel1.str.len_chars() >= 2).then(skel1 + "|" + num).alias("k_skel1_num"),
        (num + "|" + pl.col("ra1")).alias("k_num_ra1"),
        (pl.min_horizontal("ra1", "ra2") + "|" + pl.max_horizontal("ra1", "ra2")).alias("k_ra12"),
    ).select(
        "src", "rid", "country",
        # store each key as a 64-bit number instead of text: same matching, far less memory
        *[pl.when(pl.col(k).is_not_null()).then(pl.col(k).hash(seed=7)).alias(k) for k in KEYS],
    )


def add_keys(df: pl.DataFrame, dfs: pl.DataFrame) -> pl.DataFrame:
    """Keys for every record (chunked). Returns only src, rid, country + key columns."""
    return pl.concat([_add_keys(df.slice(i, CHUNK), dfs) for i in range(0, df.height, CHUNK)])


def candidates(s1k: pl.DataFrame, s23k: pl.DataFrame) -> pl.DataFrame:
    """(s1_rid, src, cand_rid) for every S1 / S2-3 pair sharing a usable key."""
    parts = []
    for k1, k2 in KEY_PAIRS:
        right = s23k.select("country", pl.col(k2).alias("k"), "src", "rid").drop_nulls("k")
        sizes = right.group_by("country", "k").len()
        right = right.join(sizes.filter(pl.col("len") <= MAX_BLOCK), on=["country", "k"]).drop("len")
        left = s1k.select("country", pl.col(k1).alias("k"), pl.col("rid").alias("s1_rid")).drop_nulls("k")
        parts.append(left.join(right, on=["country", "k"]).select("s1_rid", "src", pl.col("rid").alias("cand_rid")))
    return pl.concat(parts).unique()