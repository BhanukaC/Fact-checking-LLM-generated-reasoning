# step11b_compare_annotation.py: compare the filled annotation with the verifier per condition
import csv, json
from collections import Counter
from step8_flag import label

T = json.load(open("threshold.json"))["threshold"]
CONDITIONS = ["gold", "bm25", "similarity", "verifier"]
col = "truth (true / false / cant_tell)"

ann = {
    r["sid"]: r[col].strip().lower()
    for r in csv.DictReader(open("statement_annotation.csv", encoding="utf-8-sig"))
}
print("human labels:", dict(Counter(ann.values())))
for cond in CONDITIONS:
    vd = {
        r["item_id"]: r
        for r in map(json.loads, open(f"verdicts_{cond}_eval.jsonl", encoding="utf-8"))
    }
    by_truth = {"true": Counter(), "false": Counter()}
    for sid, truth in ann.items():
        if truth in by_truth:
            v = vd[sid]
            by_truth[truth][label(v["max_entailment"], v["max_contradiction"], T)] += 1
    f, t = by_truth["false"], by_truth["true"]
    nf, nt = sum(f.values()), sum(t.values())
    print(
        f"{cond:>10}: false statements refuted {f['refuted']}/{nf} | "
        f"true statements wrongly refuted {t['refuted']}/{nt} | "
        f"false: {dict(f)} | true: {dict(t)}"
    )
