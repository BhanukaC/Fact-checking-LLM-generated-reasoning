import csv, json, random
from collections import Counter, defaultdict

claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
}
recs = [json.loads(l) for l in open("outputs_eval.jsonl", encoding="utf-8")]

# 1. One row per (claim, model) with the right/wrong flag
rows = []
for r in recs:
    c = claims[r["claim_id"]]
    gold = c["gold_label"].upper()
    rows.append(
        {
            "claim_id": r["claim_id"],
            "model": r["model"],
            "claim": c["claim"],
            "claim_date": c["claim_date"],
            "speaker": c["speaker"] or "",
            "gold_label": gold,
            "gold_justification": c["gold_justification"],
            "llm_reasoning": r["llm_reasoning"],
            "llm_label": r["llm_label"],
            "label_correct": int(r["llm_label"] == gold),
        }
    )

# 2. Difficulty: how many of the 3 models got each claim right
right = defaultdict(int)
for row in rows:
    right[row["claim_id"]] += row["label_correct"]
for row in rows:
    row["models_right"] = right[row["claim_id"]]

rows.sort(key=lambda x: (x["claim_id"], x["model"]))
with open("dataset_eval.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# 3. Summary report
lines = []
for m in sorted({r["model"] for r in rows}):
    rs = [r for r in rows if r["model"] == m]
    wrong = [r for r in rs if not r["label_correct"]]
    lines.append(
        f"{m}: {len(rs) - len(wrong)} right, {len(wrong)} wrong "
        f"({sum(r['gold_label'] == 'SUPPORTED' for r in wrong)} on Supported, "
        f"{sum(r['gold_label'] == 'REFUTED' for r in wrong)} on Refuted)"
    )
diff = Counter(right[c] for c in {r["claim_id"] for r in rows})
lines.append("claims by number of models right: " + str(dict(sorted(diff.items()))))
lines.append(
    "claims all models got wrong: " + str(sorted(c for c, v in right.items() if v == 0))
)
open("report_eval.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# 4. Spot-check sheet: 10 wrong + 10 right outputs, fixed seed
rng = random.Random(2026)
wrong = [r for r in rows if not r["label_correct"]]
ok = [r for r in rows if r["label_correct"]]
pick = rng.sample(wrong, 10) + rng.sample(ok, 10)
with open("spot_check_eval.csv", "w", newline="", encoding="utf-8") as f:
    fields = [
        "claim_id",
        "model",
        "claim",
        "claim_date",
        "gold_label",
        "gold_justification",
        "llm_label",
        "llm_reasoning",
        "reasoning_ok (yes/no)",
        "notes",
    ]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in pick:
        w.writerow({**r, "reasoning_ok (yes/no)": "", "notes": ""})

# 5. Gold-label check sheet: claims every model got wrong
with open("gold_label_check.csv", "w", newline="", encoding="utf-8") as f:
    fields = [
        "claim_id",
        "claim",
        "claim_date",
        "gold_label",
        "gold_justification",
        "gold_label_looks_right (yes/no/unsure)",
        "notes",
    ]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for cid in sorted(c for c, v in right.items() if v == 0):
        c = claims[cid]
        w.writerow(
            {
                **c,
                "gold_label": c["gold_label"].upper(),
                "gold_label_looks_right (yes/no/unsure)": "",
                "notes": "",
            }
        )
