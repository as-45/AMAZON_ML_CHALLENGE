"""Stage 1: turn raw names and addresses into clean, comparable fields.

Adds: name_clean, name_core, name_compact, legal_form, is_web, name_nonlatin,
      addr_clean, addr_numbers, postcode, is_landmark, addr_missing
Nothing here looks at `country`, so France goes through the same code.
"""
import polars as pl
from unidecode import unidecode

# ---------- name words -> canonical form
NAME_MAP = {
    "incorporated": "inc", "incorporation": "inc",
    "limited": "ltd", "ltda": "ltd",
    "private": "pvt",
    "corporation": "corp", "corpn": "corp",
    "company": "co", "compagnie": "co", "cie": "co",
    "and": "&", "et": "&",
    "international": "intl", "internationale": "intl",
    "services": "svc", "service": "svc",
    "enterprises": "ent", "enterprise": "ent",
    "industries": "ind", "industry": "ind",
    "technologies": "tech", "technology": "tech",
    "associates": "assoc", "association": "assoc",
    "brothers": "bros",
    "establishments": "ets", "etablissements": "ets",
    # legal words after a round trip through Hindi script (very common in S2/S3 India)
    "limittedd": "ltd", "limittett": "ltd", "limirrrrdd": "ltd", "limitedd": "ltd", "li": "ltd",
    "praaivett": "pvt", "praiveett": "pvt", "piraiveett": "pvt", "praaibhett": "pvt",
    "praivrrrr": "pvt", "praivett": "pvt", "praa": "pvt", "elelpii": "llp",
}
LEGAL = {
    "inc", "llc", "ltd", "pvt", "corp", "co", "llp", "lp", "pc", "plc", "pllc",
    "opc", "pty", "gmbh", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop",
}
FILLER = {"the", "m/s", "ms", "messrs", "shri", "sri", "smt", "ets", "&"}

# ---------- address words -> full form
ADDR_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bld": "boulevard", "bd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "blk": "block", "hno": "house", "h": "house",
    "nr": "near", "opp": "opposite", "bhd": "behind",
    "ngr": "nagar", "clny": "colony", "mkt": "market", "stn": "station",
    "n": "north", "s": "south", "e": "east", "w": "west",
    # France
    "r": "rue", "all": "allee", "imp": "impasse", "rte": "route", "ch": "chemin",
    "chem": "chemin", "qu": "quai", "fbg": "faubourg", "crs": "cours", "pass": "passage",
}
LANDMARK = {"near", "opposite", "behind", "beside", "adjacent", "landmark", "pres", "face"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
PHRASES = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "rhode island": "ri",
    "district of columbia": "dc",
    "madhya pradesh": "mp", "uttar pradesh": "up", "andhra pradesh": "ap",
    "himachal pradesh": "hp", "arunachal pradesh": "arp", "tamil nadu": "tn",
    "west bengal": "wb", "jammu and kashmir": "jk",
}
IN_STATES = {
    "maharashtra": "mh", "karnataka": "ka", "gujarat": "gj", "telangana": "ts",
    "kerala": "kl", "rajasthan": "rj", "haryana": "hr", "punjab": "pb", "bihar": "br",
    "odisha": "od", "orissa": "od", "delhi": "dl", "jharkhand": "jh",
    "chhattisgarh": "cg", "uttarakhand": "uk",
}
# Only replace a state when it is a whole comma-separated part ("..., Ohio" not "Oregon Drive")
REGIONS = {**PHRASES, **US_STATES, **IN_STATES}
REGION_RE = r"(^|,)\s*(" + "|".join(sorted(REGIONS, key=len, reverse=True)) + r")\s*(,|$)"

WEB_RE = r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|co\.in|net|org|fr|biz|info|io|co)/?$"


# ---------------------------------------------------------------- helpers
# Indian-script -> English word dictionary learned from training pairs (build_dict.py).
# Empty until load_indic() is called; words not in it fall back to unidecode.
INDIC: dict = {}


def load_indic(path) -> int:
    """Load work/indic_dict.json if it exists. Returns the number of words loaded."""
    import json
    from pathlib import Path
    INDIC.clear()
    if Path(path).exists():
        INDIC.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return len(INDIC)


def _translit(s: str | None) -> str | None:
    if s is None or s.isascii():
        return s
    if INDIC:
        s = " ".join(INDIC.get(w, w) for w in s.split())
        if s.isascii():
            return s
    return unidecode(s)


def _translit_expr(col: str) -> pl.Expr:
    return pl.col(col).map_elements(_translit, return_dtype=pl.String, skip_nulls=True)


def _map_tokens(text: pl.Expr, mapping: dict) -> pl.Expr:
    """Split on spaces, map each word through `mapping`, drop empties, re-join."""
    return (
        text.str.split(" ")
        .list.eval(pl.element().replace(mapping).filter(pl.element() != ""))
        .list.join(" ")
    )


def _fix_digit_letters(e: pl.Expr) -> pl.Expr:
    """m0tors -> motors, stee1 -> steel."""
    for d, ch in (("0", "o"), ("1", "l"), ("3", "e"), ("5", "s")):
        for _ in range(2):
            e = e.str.replace_all(rf"([a-z]){d}([a-z])", rf"${{1}}{ch}${{2}}")
        e = e.str.replace_all(rf"([a-z]{{2}}){d}\b", rf"${{1}}{ch}")
    return e



def skeleton(e: pl.Expr) -> pl.Expr:
    """Sound-alike 'skeleton': drop vowels/h/y and repeated letters, merge look-alike sounds.
    'baabaa paavr' and 'baba power' both become 'b pvr'."""
    e = (
        e.str.replace_all("ph", "f").str.replace_all("x", "ks").str.replace_all("w", "v")
        .str.replace_all("[cq]", "k").str.replace_all("z", "j").str.replace_all("[aeiouhy]", "")
    )
    for ch in "bdfgjklmnprstv":
        e = e.str.replace_all(f"{ch}+", ch)
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def _replace_regions(e: pl.Expr) -> pl.Expr:
    for _ in range(2):
        e = e.str.replace_all(REGION_RE, "${1} <<${2}>> ${3}")
    for name, code in REGIONS.items():
        e = e.str.replace_all(f"<<{name}>>", code, literal=True)
    return e


# ---------- main
def normalize(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("business_name").str.contains(r"[^\x00-\x{024F}]").alias("name_nonlatin"),
        _translit_expr("business_name").alias("_n"),
        _translit_expr("business_address").alias("_a"),
    )

    # ---- names
    n = pl.col("_n").str.to_lowercase().str.strip_chars()
    df = df.with_columns(n.str.extract(WEB_RE, 1).alias("_web"))
    n = (
        pl.when(pl.col("_web").is_not_null()).then(pl.col("_web")).otherwise(n)
        .str.replace_all(r"\bm/s\b", " m/s ")
        .str.replace_all(r"['’`]", "")
        .str.replace_all(r"&", " & ")
    )
    n = _fix_digit_letters(n)
    n = n.str.replace_all(r"[^a-z0-9&/ ]", " ").str.replace_all(r"/", " ")
    n = n.str.replace_all(r"\bm s\b", "m/s").str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(
        _map_tokens(n, NAME_MAP).alias("name_clean"),
        pl.col("_web").is_not_null().alias("is_web"),
    )
    toks = pl.col("name_clean").str.split(" ")
    df = df.with_columns(
        toks.list.eval(pl.element().filter(pl.element().is_in(list(LEGAL))))
        .list.join(" ").alias("legal_form"),
        toks.list.eval(
            pl.element().filter(~pl.element().is_in(list(LEGAL | FILLER)) & (pl.element() != ""))
        ).list.join(" ").alias("name_core"),
    )
    df = df.with_columns(
        pl.when(pl.col("name_core") == "").then(pl.col("name_clean"))
        .otherwise(pl.col("name_core")).alias("name_core")
    ).with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact"),
        skeleton(pl.col("name_core")).alias("name_skel"),
    )

    # ---- addresses
    a = (
        pl.col("_a").fill_null("").str.to_lowercase()
        .str.replace_all(r"\bn\s*deg\b|\bdeg\b|#", " ")    # "N°" becomes "Ndeg" after unidecode
        .str.replace_all(r"\bno\.", " no ")
        .str.replace_all(r"['’`]", "")
        .str.replace_all(r"\s+", " ").str.strip_chars()
    )
    a = _replace_regions(a)
    a = a.str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(_map_tokens(a, ADDR_MAP).alias("addr_clean"))
    df = df.with_columns(
        pl.col("addr_clean").str.extract_all(r"\d+").list.unique(maintain_order=True).alias("addr_numbers"),
        pl.col("addr_clean").str.extract_all(r"\b\d{5,6}\b").list.last().alias("postcode"),
        pl.col("addr_clean").str.split(" ").list.eval(pl.element().is_in(list(LANDMARK)))
        .list.any().alias("is_landmark"),
        (pl.col("addr_clean") == "").alias("addr_missing"),
    )
    return df.drop("_n", "_a", "_web")


def normalize_one(name: str, address: str | None) -> dict:
    """Clean a single record (for quick testing)."""
    df = pl.DataFrame(
        {"business_name": [name], "business_address": [address]},
        schema={"business_name": pl.String, "business_address": pl.String},
    )
    return normalize(df).row(0, named=True)