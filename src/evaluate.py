# This calculates the leaderboard score (macro F0.5) on data where you know the answers.

"""Macro F0.5 exactly as the leaderboard computes it.

truth / pred: dict {s1_id: set(matched ids)}. Every S1 in `truth` is scored;
an S1 missing from `pred` counts as an empty prediction.
"""


def f05(t: set, p: set) -> float:
    if not t and not p:
        return 1.0
    if not t or not p:
        return 0.0
    hit = len(t & p)
    if hit == 0:
        return 0.0
    prec, rec = hit / len(p), hit / len(t)
    return 1.25 * prec * rec / (0.25 * prec + rec)


def macro_f05(truth: dict, pred: dict) -> float:
    return sum(f05(t, pred.get(s1, set())) for s1, t in truth.items()) / len(truth)