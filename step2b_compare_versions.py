import json
from collections import Counter

claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
}
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def load(path):
    return {
        (r["claim_id"], r["model"]): r
        for r in map(json.loads, open(path, encoding="utf-8"))
    }


v2, v3 = load("outputs_eval_v2.jsonl"), load("outputs_eval.jsonl")
#print(list(v3.items())[0])

# 1. Sanity check of the new run
print(
    "v3 outputs:",
    len(v3),
    "| invalid:",
    sum(not r["valid"] for r in v3.values()),
    "| prompt versions:",
    dict(Counter(r["prompt_version"] for r in v3.values())),
)


def ambiguous(cid):
    d, m, _ = (int(x) for x in claims[cid]["claim_date"].split("-"))
    return d <= 12 and d != m  # could be read as month-day


# 2. How labels changed between v2 and v3
changes = Counter()
for key, new in v3.items():
    old = v2.get(key)
    if not old:
        continue
    gold = claims[key[0]]["gold_label"].upper()
    group = "ambiguous date" if ambiguous(key[0]) else "clear date"
    was_ok, now_ok = old["llm_label"] == gold, new["llm_label"] == gold
    if was_ok == now_ok:
        changes[(group, "unchanged")] += 1
    else:
        changes[(group, "wrong -> right" if now_ok else "right -> wrong")] += 1

print("\nCorrectness changes, v2 -> v3:")
for (group, kind), n in sorted(changes.items()):
    print(f"  {group:15} {kind:15} {n}")

# 3. Does any v3 reasoning still state a swapped date?
left = []
for (cid, model), r in v3.items():
    d, m, _ = (int(x) for x in claims[cid]["claim_date"].split("-"))
    if (
        ambiguous(cid)
        and r["llm_reasoning"]
        and f"{MONTHS[d - 1]} {m}" in r["llm_reasoning"]
    ):
        left.append((cid, model))
print("\nv3 outputs still stating a swapped date:", left or "none")
