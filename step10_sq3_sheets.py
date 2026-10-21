# step10_sq3_sheets.py: reading sheets for the SQ3 error analysis (eval, threshold from threshold.json)
#   sq3_missed_errors.csv : wrong outputs NOT flagged, per condition (all of them)
#   sq3_false_flags.csv   : right outputs flagged, a random sample of 10 per condition
import csv, json, random
from step8_flag import label, short

CONDITIONS = ["gold", "bm25", "similarity", "verifier"]
T = json.load(open("threshold.json"))["threshold"]
N_FALSE, SEED = 10, 2026


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
    if c["split"] == "eval"
}
reasoning = {
    (o["claim_id"], o["model"]): o["llm_reasoning"] for o in jl("outputs_eval.jsonl")
}

texts, verd, evid = {}, {}, {}  # passage text, verdicts per item, passages per item
for c in CONDITIONS:
    evid[c] = {r["item_id"]: r for r in jl(f"evidence_selected_{c}_eval.jsonl")}
    for r in evid[c].values():
        for p in r["passages"]:
            texts[(c, p["eid"])] = p["text"]
    verd[c] = {r["item_id"]: r for r in jl(f"verdicts_{c}_eval.jsonl")}
flags = {
    c: {
        (int(r["claim_id"]), r["model"]): r
        for r in csv.DictReader(open(f"flags_{c}_eval.csv", encoding="utf-8"))
    }
    for c in CONDITIONS
}
gold_qa = {
    cid: "\n".join(p["text"] for p in evid["gold"][f"claim:{cid}"]["passages"])
    for cid in claims
}


def statements_of(c, cid, model):
    prefix = f"{cid}-{short(model)}-"
    items = [v for k, v in verd[c].items() if k.startswith(prefix)]
    return sorted(items, key=lambda v: int(v["item_id"].rsplit("-", 1)[1]))


def describe(c, v, all_passages):
    lab = label(v["max_entailment"], v["max_contradiction"], T)
    head = f"[{v['item_id']}] {lab.upper()} (e {v['max_entailment']:.2f}, c {v['max_contradiction']:.2f}): {v['query']}"
    if all_passages:
        ps = evid[c][v["item_id"]]["passages"]
        return head + "".join(f"\n    - {texts[(c, p['eid'])][:170]}" for p in ps)
    return (
        head
        + f"\n    strongest contradiction: {texts[(c, v['best_contradiction_eid'])][:220]}"
    )


def claim_line(c, cid):
    v = verd[c][f"claim:{cid}"]
    lab = label(v["max_entailment"], v["max_contradiction"], T)
    return (
        f"claim verdict {lab.upper()} (e {v['max_entailment']:.2f}, c {v['max_contradiction']:.2f}); "
        f"strongest contradiction: {texts[(c, v['best_contradiction_eid'])][:160]}"
    )


def base(c, key):
    cid, model = key
    f = flags[c][key]
    return {
        "claim_id": cid,
        "model": model,
        "condition": c,
        "claim": claims[cid]["claim"],
        "gold_label": f["gold_label"],
        "llm_label": f["llm_label"],
        "llm_reasoning": reasoning[key],
    }


# Sheet A: every wrong output that a condition missed
missed = []
for c in CONDITIONS:
    for key, f in sorted(flags[c].items()):
        if f["wrong"] == "True" and f["flagged"] == "False":
            missed.append(
                {
                    **base(c, key),
                    "flagged_by": " ".join(
                        x for x in CONDITIONS if flags[x][key]["flagged"] == "True"
                    )
                    or "none",
                    "gold_evidence_qa": gold_qa[key[0]],
                    "statements_and_passages": "\n".join(
                        describe(c, v, True) for v in statements_of(c, *key)
                    ),
                    "claim_check": claim_line(c, key[0]),
                    "error_statement (sid or none)": "",
                    "category (no factual error / retrieval failure / partial evidence / verifier failure)": "",
                    "notes": "",
                }
            )

# Sheet B: a random sample of right outputs that a condition flagged
false_flags = []
for c in CONDITIONS:
    rng = random.Random(f"{SEED}-{c}")
    pool = [
        key
        for key, f in sorted(flags[c].items())
        if f["wrong"] == "False" and f["flagged"] == "True"
    ]
    for key in sorted(rng.sample(pool, min(N_FALSE, len(pool)))):
        f = flags[c][key]
        refuted = set(f["refuted_sids"].split())
        reason = " + ".join(
            r
            for r, on in (
                ("statement", f["flag_statement"] == "True"),
                ("claim check", f["flag_claim"] == "True"),
            )
            if on
        )
        false_flags.append(
            {
                **base(c, key),
                "flag_reason": reason,
                "refuted_statements": "\n".join(
                    describe(c, v, False)
                    for v in statements_of(c, *key)
                    if v["item_id"] in refuted
                ),
                "claim_check": claim_line(c, key[0]),
                "type (misinformation itself / different event-place-time / framing-opinion / "
                "fragment / NLI error / real error / other)": "",
                "notes": "",
            }
        )

# Four files: gold rows (Yoke) and retrieval rows (Bhanuka), so both can work independently
OWNERS = {"gold": ["gold"], "retrieval": ["bm25", "similarity", "verifier"]}

for sheet, rows in (("missed_errors", missed), ("false_flags", false_flags)):
    for owner, conds in OWNERS.items():
        part = [r for r in rows if r["condition"] in conds]
        path = f"sq3_{sheet}_{owner}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(part)
        print(f"{path}: {len(part)} rows")
