# sq3_counts.py: count the SQ3 failure-analysis labels from the four reading sheets
#   python sq3_counts.py
import csv, os
from collections import Counter

SHEETS = [
    ("sq3_false_flags_gold.csv", "type"),
    ("sq3_false_flags_retrieval.csv", "type"),
    ("sq3_missed_errors_gold.csv", "category"),
    ("sq3_missed_errors_retrieval.csv", "category"),
]

for path, col_prefix in SHEETS:
    if not os.path.exists(path):
        print(f"\n{path}: not found")
        continue
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    col = next(k for k in rows[0] if k.startswith(col_prefix))
    labels = [r[col].strip() for r in rows]
    print(f"\n{path}: {len(rows)} rows, {sum(not l for l in labels)} unlabelled")
    for lab, n in Counter(labels).most_common():
        print(f"  {n:3}  {lab or '(empty)'}")
    conds = sorted({r["condition"] for r in rows})
    if len(conds) > 1:
        for c in conds:
            sub = Counter(r[col].strip() for r in rows if r["condition"] == c)
            print(
                f"    {c:>10}: " + ", ".join(f"{k} {v}" for k, v in sub.most_common())
            )
    if col_prefix == "category":
        real = [r for r in rows if r[col].strip() != "no factual error"]
        print(
            f"  real misses (a factual error present or unclear): {len(real)}; claims: "
            + ", ".join(sorted({r["claim_id"] for r in real}, key=int))
        )
