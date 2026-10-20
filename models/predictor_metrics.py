"""Predictor metrics, reported separately for presence, order, and parameters.

* Presence  - per-sample set precision/recall/F1 and exact set match (order ignored).
* Order     - judged on the *decoded sequences*: exact-sequence match, normalized
              Levenshtein distance, first-effect accuracy, length accuracy, and
              pairwise order accuracy (for every pair of effects present in the ground
              truth, is their relative order predicted correctly; a pair with a missing
              effect counts as wrong). ``order_given_correct_set`` isolates order skill
              from presence errors.
* Parameter - masked MAE over parameters of effects that are truly present AND predicted.
"""
from __future__ import annotations

from itertools import combinations


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sequence_metrics(preds: list[list[str]], truths: list[list[str]]) -> dict[str, float]:
    n = len(truths)
    if n == 0 or n != len(preds):
        raise ValueError("preds/truths must be equal length and non-empty")
    exact = first = length = 0
    ned = 0.0
    pair_ok = pair_tot = 0
    p_sum = r_sum = f_sum = 0.0
    set_exact = 0
    ord_correct_set = ord_correct_set_n = 0
    for p, t in zip(preds, truths):
        exact += p == t
        first += bool(p) and p[0] == t[0]
        length += len(p) == len(t)
        ned += edit_distance(p, t) / max(len(p), len(t), 1)
        sp, st = set(p), set(t)
        tp = len(sp & st)
        prec = tp / len(sp) if sp else 0.0
        rec = tp / len(st) if st else 1.0
        p_sum += prec; r_sum += rec
        f_sum += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        set_exact += sp == st
        pos = {e: i for i, e in enumerate(p)}
        for a, b in combinations(t, 2):  # a precedes b in truth
            pair_tot += 1
            pair_ok += a in pos and b in pos and pos[a] < pos[b]
        if sp == st:
            ord_correct_set_n += 1
            ord_correct_set += p == t
    out = {
        "presence_precision": p_sum / n, "presence_recall": r_sum / n, "presence_f1": f_sum / n,
        "presence_set_exact": set_exact / n,
        "order_exact_sequence": exact / n, "order_first_effect_acc": first / n,
        "order_length_acc": length / n, "order_norm_edit_distance": ned / n,
        "order_pairwise_acc": pair_ok / pair_tot if pair_tot else float("nan"),
        "order_given_correct_set": ord_correct_set / ord_correct_set_n if ord_correct_set_n else float("nan"),
        "n_pairs": pair_tot, "n": n,
    }
    return out


def parameter_mae(pred_chains: list[list[dict]], true_effects: list[list[dict]]) -> dict[str, float]:
    """MAE (normalized units) over parameters of effects present in both prediction and truth."""
    tot = cnt = 0.0
    per_effect: dict[str, list[float]] = {}
    for pc, tc in zip(pred_chains, true_effects):
        pm = {s["effect"]: s["parameters"] for s in pc}
        for step in tc:
            if step["effect"] in pm:
                truth = step.get("parameters_normalized") or step["parameters"]
                for k, v in truth.items():
                    e = abs(pm[step["effect"]][k] - v)
                    tot += e; cnt += 1
                    per_effect.setdefault(step["effect"], []).append(e)
    return {"param_mae": tot / cnt if cnt else float("nan"), "param_count": int(cnt),
            **{f"param_mae_{k}": sum(v) / len(v) for k, v in per_effect.items()}}
