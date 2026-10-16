# step8_flag.py: label statements, flag LLM outputs, choose the threshold on tune,
# and score outputs for threshold-free evaluation (AUROC, average precision)
#   python step8_flag.py --split tune   -> grid of thresholds, saves the best to threshold.json (run once)
#   python step8_flag.py --split eval   -> uses threshold.json, writes flags_<condition>_eval.csv
import argparse, csv, json, os
from collections import defaultdict

CONDITIONS = ["gold", "bm25", "similarity"]  # the threshold was chosen on these three
EXTRA = ["verifier"]  # added in Step 6b; uses the same fixed threshold
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def label(e, c, t):
    """Statement or claim label from the highest entailment (e) and contradiction (c)."""
    if c >= t and c > e:
        return "refuted"
    if e >= t:
        return "supported"
    return "unverified"


def refute_score(v):
    """Score at which a statement becomes 'refuted': refuted at t exactly when this is >= t."""
    return (
        v["max_contradiction"] if v["max_contradiction"] > v["max_entailment"] else 0.0
    )


def mismatch_score(v, llm_label):
    """Score at which the claim verdict contradicts the LLM label: mismatch at t exactly when >= t."""
    e, c = v["max_entailment"], v["max_contradiction"]
    if llm_label == "SUPPORTED":
        return c if c > e else 0.0  # claim refuted
    return e if e >= c else 0.0  # claim supported


def short(model):
    return model.split("/")[1].split("-")[
        0
    ]  # gemini / claude / gpt, as in the statement IDs


def load_outputs(split):
    gold = {
        c["claim_id"]: c["gold_label"].upper()
        for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
        if c["split"] == split
    }
    outs = []
    for line in open(f"outputs_{split}.jsonl", encoding="utf-8"):
        o = json.loads(line)
        if o["valid"]:
            outs.append(
                {
                    "claim_id": o["claim_id"],
                    "model": o["model"],
                    "short": short(o["model"]),
                    "llm_label": o["llm_label"],
                    "gold_label": gold[o["claim_id"]],
                    "wrong": o["llm_label"] != gold[o["claim_id"]],
                }
            )
    return outs


def load_verdicts(cond, split):
    rows = (
        json.loads(l) for l in open(f"verdicts_{cond}_{split}.jsonl", encoding="utf-8")
    )
    return {r["item_id"]: r for r in rows}


def flag_outputs(outs, verd, t):
    per_output = defaultdict(list)  # (claim_id, short model) -> [(sid, verdict)]
    for item_id, v in verd.items():
        if item_id.startswith("claim:"):
            continue
        cid, sh, _ = item_id.rsplit("-", 2)  # "36-gemini-2" -> "36", "gemini", "2"
        per_output[(int(cid), sh)].append((item_id, v))

    rows = []
    for o in outs:
        st = per_output.get((o["claim_id"], o["short"]), [])
        labs = [
            (sid, label(v["max_entailment"], v["max_contradiction"], t))
            for sid, v in st
        ]
        refuted = [sid for sid, lab in labs if lab == "refuted"]
        cv = verd[f"claim:{o['claim_id']}"]
        claim_verdict = {"supported": "SUPPORTED", "refuted": "REFUTED"}.get(
            label(cv["max_entailment"], cv["max_contradiction"], t), ""
        )
        claim_mismatch = claim_verdict != "" and claim_verdict != o["llm_label"]
        stmt_score = max((refute_score(v) for _, v in st), default=0.0)
        claim_score = mismatch_score(cv, o["llm_label"])
        rows.append(
            {
                **{
                    k: o[k]
                    for k in ("claim_id", "model", "llm_label", "gold_label", "wrong")
                },
                "n_fact": len(st),
                "n_supported": sum(lab == "supported" for _, lab in labs),
                "n_refuted": len(refuted),
                "n_unverified": sum(lab == "unverified" for _, lab in labs),
                "claim_verdict": claim_verdict or "unverified",
                "flag_statement": bool(refuted),
                "flag_claim": claim_mismatch,
                "flagged": bool(refuted) or claim_mismatch,
                "score": round(max(stmt_score, claim_score), 4),
                "stmt_score": round(stmt_score, 4),
                "claim_score": round(claim_score, 4),
                "refuted_sids": " ".join(refuted),
            }
        )
    return rows


def metrics(rows):
    tp = sum(r["flagged"] and r["wrong"] for r in rows)
    fp = sum(r["flagged"] and not r["wrong"] for r in rows)
    fn = sum(not r["flagged"] and r["wrong"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "flagged": tp + fp,
        "n": len(rows),
    }


def auroc(rows):
    """Probability that a random wrong output scores higher than a random right one (ties count half)."""
    pos = [r["score"] for r in rows if r["wrong"]]
    neg = [r["score"] for r in rows if not r["wrong"]]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def average_precision(rows):
    """Area under the precision-recall curve, ranking outputs by score (highest first)."""
    ranked = sorted(rows, key=lambda r: -r["score"])
    n_pos = sum(r["wrong"] for r in rows)
    hits, ap = 0, 0.0
    for i, r in enumerate(ranked, start=1):
        if r["wrong"]:
            hits += 1
            ap += hits / i
    return ap / n_pos if n_pos else float("nan")


def breakdown(rows):
    s = sum(r["flag_statement"] and not r["flag_claim"] for r in rows)
    c = sum(r["flag_claim"] and not r["flag_statement"] for r in rows)
    b = sum(r["flag_claim"] and r["flag_statement"] for r in rows)
    return f"flag reasons: statement only {s}, claim only {c}, both {b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "tune"], required=True)
    args = ap.parse_args()

    outs = load_outputs(args.split)
    conds = [
        c
        for c in CONDITIONS + EXTRA
        if os.path.exists(f"verdicts_{c}_{args.split}.jsonl")
    ]
    verdicts = {c: load_verdicts(c, args.split) for c in conds}
    n_wrong = sum(o["wrong"] for o in outs)
    print(
        f"{args.split}: {len(outs)} outputs, {n_wrong} wrong (chance AP = {n_wrong / len(outs):.3f})"
    )

    if args.split == "tune":
        if os.path.exists("threshold.json"):
            raise SystemExit(
                "threshold.json exists: the threshold is fixed. Delete it only to re-tune on purpose."
            )
        best = None
        print(
            f"{'t':>5} | " + " | ".join(f"{c:^24}" for c in CONDITIONS) + " | mean F1"
        )
        for t in GRID:
            ms = {c: metrics(flag_outputs(outs, verdicts[c], t)) for c in CONDITIONS}
            mean_f1 = sum(m["f1"] for m in ms.values()) / len(ms)
            cells = " | ".join(
                f"P{m['precision']:.2f} R{m['recall']:.2f} F{m['f1']:.2f} fl{m['flagged']:>3}"
                for m in ms.values()
            )
            print(f"{t:>5} | {cells} | {mean_f1:.3f}")
            if best is None or mean_f1 > best[1]:  # ties keep the lower threshold
                best = (t, mean_f1)
        json.dump(
            {
                "threshold": best[0],
                "chosen_on": "tune",
                "mean_f1_tune": round(best[1], 4),
                "tune_wrong_outputs": n_wrong,
                "grid": GRID,
            },
            open("threshold.json", "w"),
            indent=1,
        )
        print(
            f"chosen threshold: {best[0]} (mean F1 on tune {best[1]:.3f}), saved to threshold.json"
        )
        for c in CONDITIONS:
            rows = flag_outputs(outs, verdicts[c], best[0])
            print(
                f"{c:>10}: AUROC {auroc(rows):.3f}  AP {average_precision(rows):.3f}  | {breakdown(rows)}"
            )
        return

    if not os.path.exists("threshold.json"):
        raise SystemExit(
            "Run --split tune first: the threshold must be chosen on tune."
        )
    t = json.load(open("threshold.json"))["threshold"]
    print(f"threshold {t} (from threshold.json, chosen on tune)")
    for c in conds:
        rows = flag_outputs(outs, verdicts[c], t)
        with open(
            f"flags_{c}_{args.split}.csv", "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        m = metrics(rows)
        print(
            f"\n{c:>10}: AUROC {auroc(rows):.3f}  AP {average_precision(rows):.3f}  |  at t={t}: "
            f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
            f"flagged {m['flagged']}/{m['n']}  (tp {m['tp']}, fp {m['fp']}, fn {m['fn']})"
        )
        print(f"            {breakdown(rows)}")
        for model in sorted({r["model"] for r in rows}):
            mm = metrics([r for r in rows if r["model"] == model])
            print(
                f"            {model:28} flagged {mm['flagged']:>3}/{mm['n']}  "
                f"P {mm['precision']:.2f}  R {mm['recall']:.2f}  (wrong {mm['tp'] + mm['fn']})"
            )


if __name__ == "__main__":
    main()
