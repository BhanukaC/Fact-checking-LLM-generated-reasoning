import csv, json, os, random
from collections import Counter, defaultdict

SEED = 2026
EXTRA_GOLD_CHECK = [
    99,
    258,
]  # doubtful gold labels found in the spot check (not all models wrong)

claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
}
recs = [json.loads(l) for l in open("outputs_eval.jsonl", encoding="utf-8")]


def qa_text(c):
    # Gold question-answer evidence as readable text for the check sheets
    out = []
    for q in c.get("gold_questions", []):
        answers = []
        for a in q.get("answers", []):
            text = a.get("answer", "")
            if a.get("boolean_explanation"):
                text += f" ({a['boolean_explanation']})"
            answers.append(text)
        out.append(f"Q: {q['question']}\nA: {' | '.join(answers)}")
    return "\n\n".join(out)


def write_sheet(path, fields, rows):
    # Never overwrite a sheet that may already contain our manual answers
    if os.path.exists(path):
        print(f"skipped {path}: already exists (delete it first to rebuild)")
        return
    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as f:  # utf-8-sig: opens correctly in Excel/Numbers
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows")


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
all_wrong = sorted(c for c, v in right.items() if v == 0)
lines.append("claims by number of models right: " + str(dict(sorted(diff.items()))))
lines.append("claims all models got wrong: " + str(all_wrong))
open("report_eval.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# 4. Spot-check sheet: 10 wrong + 10 right outputs, fixed seed, with gold evidence
rng = random.Random(SEED)
wrong = [r for r in rows if not r["label_correct"]]
ok = [r for r in rows if r["label_correct"]]
pick = rng.sample(wrong, 10) + rng.sample(ok, 10)
write_sheet(
    "spot_check_eval.csv",
    [
        "claim_id",
        "model",
        "claim",
        "claim_date",
        "gold_label",
        "gold_justification",
        "gold_evidence_qa",
        "llm_label",
        "llm_reasoning",
        "reasoning_ok (yes/no/unsure)",
        "notes",
    ],
    [
        {
            **r,
            "gold_evidence_qa": qa_text(claims[r["claim_id"]]),
            "reasoning_ok (yes/no/unsure)": "",
            "notes": "",
        }
        for r in pick
    ],
)

# 5. Gold-label check sheet: claims every model got wrong, plus extra doubtful ones
check_ids = sorted(set(all_wrong) | set(EXTRA_GOLD_CHECK))
write_sheet(
    "gold_label_check.csv",
    [
        "claim_id",
        "claim",
        "claim_date",
        "gold_label",
        "gold_justification",
        "gold_evidence_qa",
        "gold_label_looks_right (yes/no/unsure)",
        "notes",
    ],
    [
        {
            **claims[cid],
            "gold_label": claims[cid]["gold_label"].upper(),
            "gold_evidence_qa": qa_text(claims[cid]),
            "gold_label_looks_right (yes/no/unsure)": "",
            "notes": "",
        }
        for cid in check_ids
    ],
)
