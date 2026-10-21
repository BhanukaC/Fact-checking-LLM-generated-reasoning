# step11_annotation_sheet.py: pick 15 eval outputs (stratified, seed 2026) and write the
# statement annotation sheet. Verifier labels are NOT shown, so the annotation is blind.
import csv, json, random
from collections import defaultdict
from step8_flag import short

SEED = 2026
DOUBTFUL = {99, 113, 196, 258, 330}  # gold-label check, excluded from the wrong pool
QUOTA = {
    "wrong": {"gemini": 3, "claude": 3, "gpt": 2},  # 8 wrong
    "right": {"gemini": 2, "claude": 2, "gpt": 3},
}  # 7 right -> 5 outputs per model


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
    if c["split"] == "eval"
}
outputs = {f"{o['claim_id']}|{o['model']}": o for o in jl("outputs_eval.jsonl")}
statements = {}
for r in jl("statements_eval.jsonl"):
    if r["valid"]:
        statements[r["output_id"]] = [s for s in r["statements"] if s["type"] == "fact"]
gold_qa = {
    r["claim_id"]: "\n".join(p["text"] for p in r["passages"])
    for r in jl("evidence_selected_gold_eval.jsonl")
    if r["item_id"].startswith("claim:")
}

# Candidate pools per group and model
pools = defaultdict(list)
for oid, o in sorted(outputs.items()):
    if not statements.get(oid):
        continue  # no fact statement (e.g. 230 Claude)
    wrong = o["llm_label"] != claims[o["claim_id"]]["gold_label"].upper()
    if wrong and o["claim_id"] in DOUBTFUL:
        continue
    pools[("wrong" if wrong else "right", short(o["model"]))].append(oid)

rng = random.Random(SEED)
picked = []
for group, quota in QUOTA.items():
    for model, n in quota.items():
        pool = pools[(group, model)]
        take = rng.sample(pool, min(n, len(pool)))
        picked += [(group, oid) for oid in take]
        print(f"{group:5} {model:6}: {len(take)} of {len(pool)} candidates")

rows = []
for group, oid in sorted(picked, key=lambda x: (outputs[x[1]]["claim_id"], x[1])):
    o, c = outputs[oid], claims[outputs[oid]["claim_id"]]
    for i, s in enumerate(statements[oid]):
        first = i == 0  # long context only on the output's first row
        rows.append(
            {
                "output_id": oid,
                "group": group,
                "claim_id": o["claim_id"],
                "model": short(o["model"]),
                "claim": c["claim"] if first else "",
                "claim_date": c["claim_date"] if first else "",
                "gold_label": c["gold_label"].upper() if first else "",
                "llm_label": o["llm_label"] if first else "",
                "llm_reasoning": o["llm_reasoning"] if first else "",
                "gold_evidence_qa": gold_qa[o["claim_id"]] if first else "",
                "sid": s["sid"],
                "statement": s["text"],
                "truth (true / false / cant_tell)": "",
                "source_or_reason": "",
            }
        )

with open("statement_annotation.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"statement_annotation.csv: {len(picked)} outputs, {len(rows)} statements")
