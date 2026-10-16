# step5b_check_statements.py: summary checks on the decomposer output (statements_*.jsonl)
import collections
import json
import os

for split in ("tune", "eval"):
    path = f"statements_{split}.jsonl"
    if not os.path.exists(path):
        print(f"== {split}: {path} not found, skipped")
        continue

    # One row per output: a later valid row replaces an earlier failed one
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["valid"] or r["output_id"] not in rows:
            rows[r["output_id"]] = r
    rs = list(rows.values())

    invalid = sum(not r["valid"] for r in rs)
    new_num = sum(bool(r["new_numbers"]) for r in rs)
    size = sum(r["size_flag"] for r in rs)
    versions = dict(collections.Counter(r["prompt_version"] for r in rs))
    types = collections.Counter(s["type"] for r in rs for s in r["statements"])

    print(
        f"== {split}: outputs {len(rs)}, invalid {invalid}, new_numbers {new_num}, "
        f"size_flag {size}, versions {versions}"
    )
    print("   statement types:", dict(types))

    # Fact statements per output, per model (goes in the report)
    for m in sorted({r["model"] for r in rs}):
        facts = [
            sum(s["type"] == "fact" for s in r["statements"])
            for r in rs
            if r["model"] == m
        ]
        zero = sum(f == 0 for f in facts)
        print(
            f"   {m}: fact statements per output mean {sum(facts) / len(facts):.1f}, "
            f"min {min(facts)}, max {max(facts)}, outputs with no fact {zero}"
        )

    # Outputs to look at by hand
    for r in rs:
        if r["new_numbers"] or not r["valid"] or r["size_flag"]:
            flags = []
            if not r["valid"]:
                flags.append("invalid")
            if r["new_numbers"]:
                flags.append(f"new numbers {r['new_numbers']}")
            if r["size_flag"]:
                flags.append(f"size {len(r['statements'])}")
            print("   CHECK", r["output_id"], ", ".join(flags))
