import csv, json, random

claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
}
rows = list(csv.DictReader(open("dataset_eval.csv", encoding="utf-8")))


def qa_text(c):
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


# Same selection as step3_build_dataset.py: same seed, same order of rows
rng = random.Random(2026)
wrong = [r for r in rows if r["label_correct"] == "0"]
ok = [r for r in rows if r["label_correct"] == "1"]
pick = rng.sample(wrong, 10) + rng.sample(ok, 10)

fields = [
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
]
with open("spot_check_eval_1.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in pick:
        w.writerow(
            {
                **r,
                "gold_evidence_qa": qa_text(claims[int(r["claim_id"])]),
                "reasoning_ok (yes/no/unsure)": "",
                "notes": "",
            }
        )
print("wrote", len(pick), "rows; first:", pick[0]["claim_id"], pick[0]["model"])
