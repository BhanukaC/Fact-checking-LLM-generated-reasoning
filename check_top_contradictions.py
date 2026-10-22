# check_top_contradictions.py: the N strongest contradictions on tune (similarity), as a sheet to judge by hand
import csv, json, sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
COND = "similarity"


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
}
outputs = {
    (o["claim_id"], o["model"].split("/")[1].split("-")[0]): o
    for o in jl("outputs_tune.jsonl")
}
pool = json.load(open("evidence_clean.json", encoding="utf-8"))
sources = {
    s["sid"]: s for v in pool.values() for s in v["sentences"]
}  # passage id -> text and URL

verdicts = [
    v
    for v in jl(f"verdicts_{COND}_tune.jsonl")
    if not v["item_id"].startswith("claim:")
]
top = sorted(verdicts, key=lambda v: -v["max_contradiction"])[:N]

rows = []
for rank, v in enumerate(top, start=1):
    cid, model, _ = v["item_id"].rsplit("-", 2)
    o, c = outputs[(int(cid), model)], claims[int(cid)]
    src = sources[v["best_contradiction_eid"]]
    rows.append(
        {
            "rank": rank,
            "sid": v["item_id"],
            "contradiction": v["max_contradiction"],
            "claim": c["claim"],
            "claim_date": c["claim_date"],
            "gold_label": c["gold_label"].upper(),
            "llm_label": o["llm_label"],
            "llm_right": o["llm_label"] == c["gold_label"].upper(),
            "statement": v["query"],
            "passage": src["sentence"],
            "passage_url": src["url"],
            "real_contradiction (yes / no / contested)": "",
            "type (misinformation itself / different event / framing-opinion / fragment / NLI error)": "",
            "notes": "",
        }
    )

with open("top_contradictions_tune.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(
        f"\n#{r['rank']} [{r['contradiction']:.4f}] {r['sid']} | LLM {'right' if r['llm_right'] else 'WRONG'}"
    )
    print(f"  STATEMENT: {r['statement']}")
    print(f"  PASSAGE:   {r['passage']}")
    print(f"  SOURCE:    {r['passage_url']}")
print(f"\nWrote top_contradictions_tune.csv ({len(rows)} rows)")
