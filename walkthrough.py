# walkthrough.py: follow one LLM output through the whole pipeline
#   python walkthrough.py 36 claude              all evidence conditions
#   python walkthrough.py 36 claude similarity   one condition
import json, os, sys
from step8_flag import label, short

T = json.load(open("threshold.json"))["threshold"]
cid, sh = int(sys.argv[1]), sys.argv[2]
want = sys.argv[3:] or ["gold", "bm25", "similarity", "verifier"]


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


claim = next(
    c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
    if c["claim_id"] == cid
)
split = claim["split"]
out = next(
    o
    for o in jl(f"outputs_{split}.jsonl")
    if o["claim_id"] == cid and short(o["model"]) == sh
)
wrong = out["llm_label"] != claim["gold_label"].upper()

print(
    f"\n=== 1. CLAIM {cid} ({split})\n{claim['claim']}\ndate {claim['claim_date']} | speaker {claim['speaker']} | GOLD {claim['gold_label'].upper()}"
)
print(
    f"\n=== 2. LLM OUTPUT ({out['model']})\nlabel {out['llm_label']} -> {'WRONG' if wrong else 'right'}\n{out['llm_reasoning']}"
)

rec = [
    r
    for r in jl(f"statements_{split}.jsonl")
    if r["output_id"] == f"{cid}|{out['model']}" and r["valid"]
][-1]
print("\n=== 3. STATEMENTS (only fact statements go to the verifier)")
for s in rec["statements"]:
    print(f"  [{s['type']:5}] {s['sid']}: {s['text']}")
facts = [s["sid"] for s in rec["statements"] if s["type"] == "fact"]

for cond in want:
    ev_path, vd_path = (
        f"evidence_selected_{cond}_{split}.jsonl",
        f"verdicts_{cond}_{split}.jsonl",
    )
    if not (os.path.exists(ev_path) and os.path.exists(vd_path)):
        print(f"\n=== {cond.upper()}: not run on {split}, skipped")
        continue
    ev = {r["item_id"]: r for r in jl(ev_path)}
    vd = {r["item_id"]: r for r in jl(vd_path)}
    print(f"\n=== 4-6. CONDITION {cond.upper()}  (threshold {T})")
    labels = {}
    for item in [f"claim:{cid}"] + facts:
        v = vd[item]
        lab = label(v["max_entailment"], v["max_contradiction"], T)
        labels[item] = lab
        print(
            f"\n  {item} -> {lab.upper()}  (max entail {v['max_entailment']:.3f}, max contra {v['max_contradiction']:.3f})"
        )
        print(f"  query: {v['query']}")
        for p, pv in zip(ev[item]["passages"], v["passages"]):
            print(
                f"    e {pv['entailment']:.2f} n {pv['neutral']:.2f} c {pv['contradiction']:.2f} | {p['text'][:150]}"
            )
    refuted = [s for s in facts if labels[s] == "refuted"]
    claim_lab = labels[f"claim:{cid}"]
    mismatch = (
        claim_lab in ("supported", "refuted") and claim_lab.upper() != out["llm_label"]
    )
    flagged = bool(refuted) or mismatch
    verdict = (
        ("CAUGHT the error" if wrong else "FALSE FLAG")
        if flagged
        else ("MISSED the error" if wrong else "correctly not flagged")
    )
    print(
        f"\n  FLAG: refuted facts {refuted or 'none'} | claim verdict {claim_lab} vs LLM {out['llm_label']} "
        f"-> flagged {flagged} -> {verdict}"
    )
