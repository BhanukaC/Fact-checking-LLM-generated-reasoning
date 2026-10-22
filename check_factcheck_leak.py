# check_factcheck_leak.py: find fact-check sentences that passed the cleaning (fact-check sections of
# general news sites), then recompute AUROC without the affected eval claims.
#   python check_factcheck_leak.py
import csv, json
from collections import Counter
from step8_flag import auroc

SECTION_NAMES = [
    "webqoof",
    "hoaxposed",
]  # The Quint's and ThePrint's fact-check sections
CONDS = ["gold", "bm25", "similarity", "verifier"]

# 1. Which kept sentences come from a fact-check section?
pool = json.load(open("evidence_clean.json", encoding="utf-8"))
hits = [
    (int(cid), c["split"], s["url"], s["sentence"])
    for cid, c in pool.items()
    for s in c["sentences"]
    if any(n in s["url"].lower() for n in SECTION_NAMES)
]
total = sum(len(c["sentences"]) for c in pool.values())
print(f"leaked sentences: {len(hits)} of {total} ({len(hits) / total:.2%})")
for (cid, split), n in sorted(Counter((h[0], h[1]) for h in hits).items()):
    print(f"  claim {cid} ({split}): {n}")
for cid, split, url, sent in hits:
    print(f"    {cid}: {sent[:90]}  [{url[:70]}]")

# 2. AUROC with and without the affected eval claims
affected = {h[0] for h in hits if h[1] == "eval"}
print(f"\naffected eval claims: {sorted(affected)}")
for c in CONDS:
    rows = list(csv.DictReader(open(f"flags_{c}_eval.csv", encoding="utf-8")))
    sub = [r for r in rows if int(r["claim_id"]) not in affected]
    sc = lambda rs, f: [
        {"wrong": r["wrong"] == "True", "score": float(r[f])} for r in rs
    ]
    a, b = auroc(sc(rows, "score")), auroc(sc(sub, "score"))
    sa, sb = auroc(sc(rows, "stmt_score")), auroc(sc(sub, "stmt_score"))
    print(
        f"  {c:>10}: AUROC {a:.3f} -> {b:.3f} ({b - a:+.3f})   statements only {sa:.3f} -> {sb:.3f} ({sb - sa:+.3f})"
        f"   [{len(sub)} outputs, {sum(r['wrong'] == 'True' for r in sub)} wrong]"
    )
