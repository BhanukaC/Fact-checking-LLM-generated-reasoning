# step9_evaluate.py: AUROC/AP with claim-clustered bootstrap CIs, statement/claim split,
# paired differences, threshold metrics with clustered CIs (McNemar kept as exploratory),
# robustness set, and a descriptive threshold-sensitivity table.
#   python step9_evaluate.py | tee results_eval.txt
import csv, json, math, os, random
from step8_flag import auroc, average_precision

CONDITIONS = ["gold", "bm25", "similarity", "verifier"]
PAIRS = [
    ("gold", "bm25"),
    ("gold", "similarity"),
    ("similarity", "bm25"),
    ("verifier", "similarity"),
    ("verifier", "bm25"),
    ("gold", "verifier"),
]
DOUBTFUL = {99, 113, 196, 258, 330}  # gold-label check, rule fixed on 15 Oct
SENSITIVITY = [
    0.5,
    0.7,
    0.9,
    0.95,
    0.99,
]  # shown on eval for description only, never used to choose
B, SEED = 2000, 2026


def load(cond):
    rows = []
    for r in csv.DictReader(open(f"flags_{cond}_eval.csv", encoding="utf-8")):
        rows.append(
            {
                "claim_id": int(r["claim_id"]),
                "model": r["model"],
                "wrong": r["wrong"] == "True",
                "score": float(r["score"]),
                "stmt_score": float(r["stmt_score"]),
                "claim_score": float(r["claim_score"]),
            }
        )
    return sorted(rows, key=lambda r: (r["claim_id"], r["model"]))


def scored(rows, field):
    return [{"wrong": r["wrong"], "score": r[field]} for r in rows]


def at_threshold(rows, t):
    """Flagged at t exactly when score >= t (same rule as step 8). Returns P, R, F1, accuracy, flagged."""
    tp = sum(r["score"] >= t and r["wrong"] for r in rows)
    fp = sum(r["score"] >= t and not r["wrong"] for r in rows)
    fn = sum(r["score"] < t and r["wrong"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    acc = sum((r["score"] >= t) == r["wrong"] for r in rows) / len(rows)
    return p, rc, f1, acc, tp + fp


def ci(xs):
    xs = sorted(x for x in xs if not math.isnan(x))
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


METRICS = {
    "auroc": lambda rs, t: auroc(scored(rs, "score")),
    "ap": lambda rs, t: average_precision(scored(rs, "score")),
    "auroc_stmt": lambda rs, t: auroc(scored(rs, "stmt_score")),
    "f1": lambda rs, t: at_threshold(rs, t)[2],
    "acc": lambda rs, t: at_threshold(rs, t)[3],
}


def bootstrap(data, conds, t):
    """Resample claims (a claim's three outputs stay together); the same resamples for every
    condition and metric, so paired differences are valid."""
    claims = sorted({r["claim_id"] for r in data[conds[0]]})
    by_claim = {
        c: {k: [r for r in data[c] if r["claim_id"] == k] for k in claims}
        for c in conds
    }
    rng = random.Random(SEED)
    out = {c: {m: [] for m in METRICS} for c in conds}
    for _ in range(B):
        pick = [rng.choice(claims) for _ in claims]
        for c in conds:
            rows = [r for k in pick for r in by_claim[c][k]]
            for m, fn in METRICS.items():
                out[c][m].append(fn(rows, t))
    return out


def paired(boot, data, a, b, m, t):
    point = METRICS[m](data[a], t) - METRICS[m](data[b], t)
    lo, hi = ci([x - y for x, y in zip(boot[a][m], boot[b][m])])
    return point, lo, hi


def mcnemar(a, b, t):
    ca = [(r["score"] >= t) == r["wrong"] for r in a]
    cb = [(r["score"] >= t) == r["wrong"] for r in b]
    n01 = sum(x and not y for x, y in zip(ca, cb))
    n10 = sum(y and not x for x, y in zip(ca, cb))
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return n01, n10, p


def main():
    t = json.load(open("threshold.json"))["threshold"]
    conds = [c for c in CONDITIONS if os.path.exists(f"flags_{c}_eval.csv")]
    data = {c: load(c) for c in conds}
    keys = [(r["claim_id"], r["model"]) for r in data[conds[0]]]
    assert all(
        [(r["claim_id"], r["model"]) for r in data[c]] == keys for c in conds
    ), "outputs do not line up"
    n_wrong = sum(r["wrong"] for r in data[conds[0]])
    print(
        f"eval: {len(keys)} outputs, {n_wrong} wrong, chance AP {n_wrong / len(keys):.3f}, "
        f"threshold {t} (chosen on tune, 3 wrong of 60), bootstrap {B} over claims (seed {SEED})"
    )
    boot = bootstrap(data, conds, t)

    print("\n1. Main results (threshold-free; 95% CI)")
    for c in conds:
        a, ap = METRICS["auroc"](data[c], t), METRICS["ap"](data[c], t)
        (al, ah), (pl, ph) = ci(boot[c]["auroc"]), ci(boot[c]["ap"])
        print(
            f"  {c:>10}: AUROC {a:.3f} [{al:.3f}, {ah:.3f}]  AP {ap:.3f} [{pl:.3f}, {ph:.3f}]"
        )

    print("\n2. Statements only (95% CI) vs claim check only")
    for c in conds:
        lo, hi = ci(boot[c]["auroc_stmt"])
        print(
            f"  {c:>10}: statements {METRICS['auroc_stmt'](data[c], t):.3f} [{lo:.3f}, {hi:.3f}]   "
            f"claim check {auroc(scored(data[c], 'claim_score')):.3f}"
        )

    print("\n3. Paired differences, first minus second (95% CI)")
    for a, b in PAIRS:
        if a in conds and b in conds:
            r1, r2, r3 = (
                paired(boot, data, a, b, m, t) for m in ("auroc", "ap", "auroc_stmt")
            )
            print(
                f"  {a:>10} - {b:<10}: dAUROC {r1[0]:+.3f} [{r1[1]:+.3f}, {r1[2]:+.3f}]  "
                f"dAP {r2[0]:+.3f} [{r2[1]:+.3f}, {r2[2]:+.3f}]  dAUROC stmt {r3[0]:+.3f} [{r3[1]:+.3f}, {r3[2]:+.3f}]"
            )

    print(
        f"\n4. At the fixed threshold t={t}: claim-clustered CIs (main) and McNemar (exploratory, assumes independent outputs)"
    )
    for c in conds:
        p, r, f1, acc, fl = at_threshold(data[c], t)
        print(
            f"  {c:>10}: P {p:.3f} R {r:.3f} F1 {f1:.3f} accuracy {acc:.3f} flagged {fl}"
        )
    for a, b in PAIRS:
        if a in conds and b in conds:
            da, df = paired(boot, data, a, b, "acc", t), paired(
                boot, data, a, b, "f1", t
            )
            n01, n10, p = mcnemar(data[a], data[b], t)
            print(
                f"  {a:>10} - {b:<10}: dAccuracy {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}]  "
                f"dF1 {df[0]:+.3f} [{df[1]:+.3f}, {df[2]:+.3f}]  | McNemar n01 {n01} n10 {n10} p {p:.4f}"
            )

    print(f"\n5. Robustness: without the doubtful-label claims {sorted(DOUBTFUL)}")
    for c in conds:
        rows = [r for r in data[c] if r["claim_id"] not in DOUBTFUL]
        p, r, f1, acc, fl = at_threshold(rows, t)
        print(
            f"  {c:>10}: {len(rows)} outputs, {sum(x['wrong'] for x in rows)} wrong | "
            f"AUROC {METRICS['auroc'](rows, t):.3f}  AP {METRICS['ap'](rows, t):.3f}  "
            f"statements only {METRICS['auroc_stmt'](rows, t):.3f} | P {p:.3f} R {r:.3f} F1 {f1:.3f} flagged {fl}"
        )

    print("\n6. Threshold sensitivity on eval (descriptive only; t was fixed on tune)")
    print("     t | " + " | ".join(f"{c:^26}" for c in conds))
    for tt in SENSITIVITY:
        cells = []
        for c in conds:
            p, r, f1, acc, fl = at_threshold(data[c], tt)
            cells.append(f"P {p:.2f} R {r:.2f} F1 {f1:.2f} fl {fl:>3}")
        print(f"  {tt:>4} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
