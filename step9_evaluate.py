# step9_evaluate.py: confidence intervals, paired comparisons, McNemar tests,
# robustness set and the statement/claim split. Reads flags_<condition>_eval.csv.
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
B, SEED = 2000, 2026


def load(cond):
    rows = []
    for r in csv.DictReader(open(f"flags_{cond}_eval.csv", encoding="utf-8")):
        rows.append(
            {
                "claim_id": int(r["claim_id"]),
                "model": r["model"],
                "wrong": r["wrong"] == "True",
                "flagged": r["flagged"] == "True",
                "score": float(r["score"]),
                "stmt_score": float(r["stmt_score"]),
                "claim_score": float(r["claim_score"]),
            }
        )
    return sorted(rows, key=lambda r: (r["claim_id"], r["model"]))


def scored(rows, field):
    return [{"wrong": r["wrong"], "score": r[field]} for r in rows]


def prf(rows):
    tp = sum(r["flagged"] and r["wrong"] for r in rows)
    fp = sum(r["flagged"] and not r["wrong"] for r in rows)
    fn = sum(not r["flagged"] and r["wrong"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0), tp + fp


def ci(xs):
    xs = sorted(x for x in xs if not math.isnan(x))
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


def bootstrap(data, conds):
    """Resample claims (all three outputs of a claim together), the same resamples for every condition."""
    claims = sorted({r["claim_id"] for r in data[conds[0]]})
    by_claim = {
        cond: {c: [r for r in data[cond] if r["claim_id"] == c] for c in claims}
        for cond in conds
    }
    rng = random.Random(SEED)
    out = {cond: {"auroc": [], "ap": []} for cond in conds}
    for _ in range(B):
        pick = [rng.choice(claims) for _ in claims]
        for cond in conds:
            rows = scored([r for c in pick for r in by_claim[cond][c]], "score")
            out[cond]["auroc"].append(auroc(rows))
            out[cond]["ap"].append(average_precision(rows))
    return out


def mcnemar(a, b):
    """Exact McNemar test on per-output correctness (flag decision == wrong) at the fixed threshold."""
    ca = [r["flagged"] == r["wrong"] for r in a]
    cb = [r["flagged"] == r["wrong"] for r in b]
    n01 = sum(x and not y for x, y in zip(ca, cb))  # a right, b wrong
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
        f"threshold {t}, bootstrap {B} (seed {SEED})"
    )

    boot = bootstrap(data, conds)
    print("\n1. Main results (95% bootstrap CI over claims)")
    for c in conds:
        rows = data[c]
        a, ap = auroc(scored(rows, "score")), average_precision(scored(rows, "score"))
        (al, ah), (pl, ph) = ci(boot[c]["auroc"]), ci(boot[c]["ap"])
        p, r, f1, fl = prf(rows)
        print(
            f"  {c:>10}: AUROC {a:.3f} [{al:.3f}, {ah:.3f}]  AP {ap:.3f} [{pl:.3f}, {ph:.3f}]  "
            f"| t={t}: P {p:.3f} R {r:.3f} F1 {f1:.3f} flagged {fl}"
        )

    print(
        "\n2. Where the signal comes from: AUROC of statement scores only vs claim check only"
    )
    for c in conds:
        print(
            f"  {c:>10}: statements {auroc(scored(data[c], 'stmt_score')):.3f}   "
            f"claim check {auroc(scored(data[c], 'claim_score')):.3f}"
        )

    print("\n3. Paired differences (first minus second), 95% bootstrap CI")
    for a_, b_ in PAIRS:
        if a_ in conds and b_ in conds:
            da = [x - y for x, y in zip(boot[a_]["auroc"], boot[b_]["auroc"])]
            dp = [x - y for x, y in zip(boot[a_]["ap"], boot[b_]["ap"])]
            point_a = auroc(scored(data[a_], "score")) - auroc(
                scored(data[b_], "score")
            )
            point_p = average_precision(scored(data[a_], "score")) - average_precision(
                scored(data[b_], "score")
            )
            (l1, h1), (l2, h2) = ci(da), ci(dp)
            print(
                f"  {a_:>10} - {b_:<10}: dAUROC {point_a:+.3f} [{l1:+.3f}, {h1:+.3f}]  "
                f"dAP {point_p:+.3f} [{l2:+.3f}, {h2:+.3f}]"
            )

    print(
        f"\n4. McNemar tests at t={t} (per-output correctness; n01 = first right and second wrong)"
    )
    for a_, b_ in PAIRS:
        if a_ in conds and b_ in conds:
            n01, n10, p = mcnemar(data[a_], data[b_])
            print(f"  {a_:>10} vs {b_:<10}: n01 {n01:>3}  n10 {n10:>3}  p = {p:.4f}")

    print(f"\n5. Robustness: without the doubtful-label claims {sorted(DOUBTFUL)}")
    for c in conds:
        rows = [r for r in data[c] if r["claim_id"] not in DOUBTFUL]
        p, r, f1, fl = prf(rows)
        print(
            f"  {c:>10}: {len(rows)} outputs, {sum(x['wrong'] for x in rows)} wrong | "
            f"AUROC {auroc(scored(rows, 'score')):.3f}  AP {average_precision(scored(rows, 'score')):.3f} | "
            f"P {p:.3f} R {r:.3f} F1 {f1:.3f} flagged {fl}"
        )


if __name__ == "__main__":
    main()
