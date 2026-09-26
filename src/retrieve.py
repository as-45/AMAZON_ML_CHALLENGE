"""Step 2: similarity search (TF-IDF nearest neighbours), one country at a time.

Each record becomes two TF-IDF vectors:
    name : 4-letter pieces of the compact name   ("celinasmotors" -> "celi","elin","lina",...)
    addr : address words                          ("273","oregon","drive","xenia","oh")
Similarity of two records = average of name cosine and address cosine (0..1).

Search for one S1 record, in two passes:
  1. probe   : sparse matrix product using only the query's RARER pieces/words (seen in
               <= MAX_POSTINGS records). Common pieces ("ing", "road") would touch millions of
               records per query and make the search far too slow; rare ones carry the signal.
               Returns the DEPTH best records per query.
  2. rescore : exact name / address cosines for those records; keep the union of
               top-K by combined score, top-K_NAME by name, top-K_ADDR by address.
Plus a name-only search among records WITHOUT an address (they get half credit in the
combined score and would otherwise be missed).

Needs: scikit-learn, scipy, sparse_dot_topn (Apache-2.0).
"""
import gc

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

K, K_NAME, K_ADDR, K_EMPTY = 20, 5, 5, 5
DEPTH = 300
MAX_POSTINGS = 50_000
Q_CHUNK = 20_000


def _vectorizers():
    name = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), min_df=2, dtype=np.float32,
                           sublinear_tf=True, lowercase=False)
    addr = TfidfVectorizer(analyzer=lambda s: s.split(), min_df=2, dtype=np.float32,
                           sublinear_tf=True, lowercase=False)
    return name, addr


def _rowdot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray, step=2_000_000):
    """Cosine of row ia[k] of A with row ib[k] of B (rows are L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), step):
        e = s + step
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


def _topk(q_idx, p_idx, score, k):
    """Indices (into the arrays) of the top-k scores per query."""
    order = np.lexsort((-score, q_idx))
    q = q_idx[order]
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    rank = np.arange(len(q)) - np.repeat(starts, np.diff(np.r_[starts, len(q)]))
    return order[rank < k]


def search(c1: pl.DataFrame, c23: pl.DataFrame, threads: int = -1, extra: pl.DataFrame = None) -> pl.DataFrame:
    """(s1_rid, src, cand_rid, cos_name, cos_addr, cos_comb) candidate pairs for one country.
    `extra`: optional (s1_rid, src, cand_rid) pairs found another way (the blocking keys);
    they are added to the result with their cosines computed too."""
    from sparse_dot_topn import sp_matmul_topn

    name_txt = pl.concat([c1["name_compact"], c23["name_compact"]]).fill_null("").to_list()
    addr_txt = pl.concat([c1["addr_clean"], c23["addr_clean"]]).fill_null("").to_list()
    vn, va = _vectorizers()
    N = vn.fit_transform(name_txt).tocsr()
    A = va.fit_transform(addr_txt).tocsr()
    del name_txt, addr_txt
    n1 = c1.height
    Nq, Np = N[:n1], N[n1:]
    Aq, Ap = A[:n1], A[n1:]
    del N, A
    gc.collect()

    # combined vector: [name, addr] / sqrt(2)  -> dot product = average of the two cosines
    Pc = (sp.hstack([Np, Ap]).tocsr() * np.float32(1 / np.sqrt(2)))
    PcT = Pc.T.tocsr()
    del Pc
    # query side: drop pieces/words that are too common (prefix filtering)
    df = np.diff(PcT.indptr)
    rare = sp.diags((df <= MAX_POSTINGS).astype(np.float32))
    Qc = (sp.hstack([Nq, Aq]).tocsr() * np.float32(1 / np.sqrt(2))) @ rare

    empty_p = np.flatnonzero(c23["addr_missing"].to_numpy())
    NeT = Np[empty_p].T.tocsr()
    dfn = np.diff(NeT.indptr)
    Nq_rare = Nq @ sp.diags((dfn <= MAX_POSTINGS).astype(np.float32))

    out = []
    for s in range(0, n1, Q_CHUNK):
        e = min(s + Q_CHUNK, n1)
        R = sp_matmul_topn(Qc[s:e], PcT, top_n=DEPTH, n_threads=threads).tocoo()
        qi, pj = R.row.astype(np.int64) + s, R.col.astype(np.int64)
        cn = _rowdot(Nq, Np, qi, pj)
        ca = _rowdot(Aq, Ap, qi, pj)
        cc = (cn + ca) / 2
        keep = np.unique(np.concatenate([_topk(qi, pj, cc, K), _topk(qi, pj, cn, K_NAME),
                                         _topk(qi, pj, ca, K_ADDR)]))
        parts = [(qi[keep], pj[keep], cn[keep], ca[keep], cc[keep])]
        if len(empty_p):
            R2 = sp_matmul_topn(Nq_rare[s:e], NeT, top_n=K_EMPTY, n_threads=threads).tocoo()
            q2 = R2.row.astype(np.int64) + s
            p2 = empty_p[R2.col]
            c2 = _rowdot(Nq, Np, q2, p2)
            parts.append((q2, p2, c2, np.zeros_like(c2), c2 / 2))
        out.append(pl.DataFrame({
            "qi": np.concatenate([p[0] for p in parts]).astype(np.int32),
            "pj": np.concatenate([p[1] for p in parts]).astype(np.int32),
            "cos_name": np.concatenate([p[2] for p in parts]),
            "cos_addr": np.concatenate([p[3] for p in parts]),
            "cos_comb": np.concatenate([p[4] for p in parts]),
        }))
    if extra is not None and extra.height:
        # map (s1_rid) -> query row and (src, cand_rid) -> pool row, then compute cosines
        qmap = pl.DataFrame({"s1_rid": c1["rid"], "qi": np.arange(n1, dtype=np.int32)})
        pmap = pl.DataFrame({"src": c23["src"], "cand_rid": c23["rid"], "pj": np.arange(c23.height, dtype=np.int32)})
        ex = extra.join(qmap, on="s1_rid").join(pmap, on=["src", "cand_rid"])
        qi, pj = ex["qi"].to_numpy().astype(np.int64), ex["pj"].to_numpy().astype(np.int64)
        cn, ca = _rowdot(Nq, Np, qi, pj), _rowdot(Aq, Ap, qi, pj)
        out.append(pl.DataFrame({"qi": qi.astype(np.int32), "pj": pj.astype(np.int32),
                                 "cos_name": cn, "cos_addr": ca, "cos_comb": (cn + ca) / 2}))
    res = pl.concat(out).unique(subset=["qi", "pj"], keep="first")
    s1_rid = c1["rid"].to_numpy()
    p_src, p_rid = c23["src"].to_numpy(), c23["rid"].to_numpy()
    qi, pj = res["qi"].to_numpy(), res["pj"].to_numpy()
    return pl.DataFrame({
        "s1_rid": s1_rid[qi], "src": p_src[pj], "cand_rid": p_rid[pj],
        "cos_name": res["cos_name"], "cos_addr": res["cos_addr"], "cos_comb": res["cos_comb"],
    })