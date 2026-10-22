# check_numbers.py: recompute every number quoted in the project doc from the files, and compare
import csv, json
from collections import Counter
from step8_flag import auroc, average_precision, short

DOUBTFUL = {99, 113, 196, 258, 330}
CONDS = ["gold", "bm25", "similarity", "verifier"]
results = []


def check(name, doc, now, tol=None):
    if tol is not None:
        ok, shown = abs(doc - now) <= tol, f"{now:.3f}"
    else:
        ok, shown = doc == now, str(now)
    results.append(ok)
    print(f"{'OK  ' if ok else 'DIFF'}  {name:<58} doc {str(doc):<10} now {shown}")


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def section(title):
    print(f"\n== {title}")


# Stage 0: claims
section("Stage 0: claims")
dev = json.load(open("dev.json", encoding="utf-8"))
labs = Counter(d["label"] for d in dev)
check("AVeriTeC dev claims", 500, len(dev))
check("dev Refuted", 305, labs["Refuted"])
check("dev Supported", 122, labs["Supported"])
claims = json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
gold = {c["claim_id"]: c["gold_label"].upper() for c in claims}
for split, n in (("eval", 100), ("tune", 20)):
    cs = [c for c in claims if c["split"] == split]
    check(f"{split} claims", n, len(cs))
    check(
        f"{split} Supported",
        n // 2,
        sum(c["gold_label"].upper() == "SUPPORTED" for c in cs),
    )

# Stage 1-2: outputs and ground truth
section("Stages 1-2: LLM outputs and ground truth")
outs = jl("outputs_eval.jsonl")
check("eval outputs", 300, len(outs))
check("tune outputs", 60, len(jl("outputs_tune.jsonl")))
check("eval invalid outputs", 0, sum(not o["valid"] for o in outs))
wrong = {
    (o["claim_id"], o["model"]) for o in outs if o["llm_label"] != gold[o["claim_id"]]
}
check("wrong outputs", 32, len(wrong))
for m, right in (("gemini", 89), ("claude", 89), ("gpt", 90)):
    check(
        f"right outputs {m} (of 100)",
        right,
        sum(
            1
            for o in outs
            if short(o["model"]) == m and (o["claim_id"], o["model"]) not in wrong
        ),
    )
right_per_claim = Counter()
for o in outs:
    right_per_claim[o["claim_id"]] += (o["claim_id"], o["model"]) not in wrong
diff = Counter(right_per_claim[c["claim_id"]] for c in claims if c["split"] == "eval")
for k, v in ((0, 4), (1, 7), (2, 6), (3, 83)):
    check(f"claims with {k} models right", v, diff[k])
rob = [o for o in outs if o["claim_id"] not in DOUBTFUL]
check("robustness set outputs", 285, len(rob))
check(
    "robustness set wrong", 20, sum((o["claim_id"], o["model"]) in wrong for o in rob)
)

section("Stage 2: manual checks")
sc = list(csv.DictReader(open("spot_check_eval.csv", encoding="utf-8-sig")))
col = [k for k in sc[0] if k.startswith("reasoning_ok")][0]
cnt = Counter((r["llm_label"] != r["gold_label"], r[col].strip().lower()) for r in sc)
check("spot check rows", 20, len(sc))
check("flagged wrong, reasoning sound (yes)", 4, cnt[(True, "yes")])
check("flagged wrong, real error (no)", 6, cnt[(True, "no")])
check("flagged right, sound (yes)", 9, cnt[(False, "yes")])
check("flagged right, unsure", 1, cnt[(False, "unsure")])
gl = list(csv.DictReader(open("gold_label_check.csv", encoding="utf-8-sig")))
gcol = [k for k in gl[0] if k.startswith("gold_label_looks_right")][0]
gcnt = Counter(r[gcol].strip().lower() for r in gl)
check("gold-label check: looks right", 1, gcnt["yes"])
check("gold-label check: doubtful", 5, gcnt["no"])

# Stage 3: evidence
section("Stage 3: evidence pool")
rem = Counter(
    r["reason"] for r in csv.DictReader(open("evidence_removed.csv", encoding="utf-8"))
)
for reason, n in (
    ("duplicate", 1040),
    ("factcheck_source", 322),
    ("after_claim_date", 95),
    ("noise", 25),
    ("copy_of_claim", 12),
    ("claim_own_source", 8),
):
    check(f"removed: {reason}", n, rem[reason])
pool = json.load(open("evidence_clean.json", encoding="utf-8"))
counts = sorted(len(v["sentences"]) for v in pool.values())
check("sentences kept", 10498, sum(counts))
check("kept per claim, median", 90, counts[len(counts) // 2])
check("kept per claim, min", 54, counts[0])

# Stage 4: statements
section("Stage 4: statements")
for split, facts in (("eval", 1612), ("tune", 350)):
    latest = {}
    for r in jl(f"statements_{split}.jsonl"):
        if r["valid"]:
            latest[r["output_id"]] = r
    types = Counter(s["type"] for r in latest.values() for s in r["statements"])
    check(f"{split} fact statements", facts, types["fact"])
    if split == "eval":
        check("eval meta statements", 250, types["meta"])
        check("eval hedge statements", 71, types["hedge"])
        for m, mean in (("claude", 6.6), ("gemini", 4.7), ("gpt", 4.8)):
            fs = [
                sum(s["type"] == "fact" for s in r["statements"])
                for r in latest.values()
                if short(r["model"]) == m
            ]
            check(f"facts per output, {m}", mean, sum(fs) / len(fs), tol=0.05)

# Stage 5: evidence selection
section("Stage 5: evidence selection")
ev = {
    c: {r["item_id"]: r for r in jl(f"evidence_selected_{c}_eval.jsonl")} for c in CONDS
}
for c in CONDS:
    check(f"items, {c}", 1712, len(ev[c]))
gp = [len(r["passages"]) for k, r in ev["gold"].items() if k.startswith("claim:")]
check("gold passages per claim, mean", 2.6, sum(gp) / len(gp), tol=0.05)
check("gold passages per claim, min", 1, min(gp))
check("gold passages per claim, max", 10, max(gp))
ov = [
    len(
        {p["eid"] for p in ev["verifier"][k]["passages"]}
        & {p["eid"] for p in ev["similarity"][k]["passages"]}
    )
    for k in ev["verifier"]
]
check(
    "verifier passages shared with similarity (of 5)", 1.6, sum(ov) / len(ov), tol=0.05
)

# Stage 6: verifier
section("Stage 6: verifier (items with probability >= 0.5)")
for c, con, ent in (
    ("gold", 317, 478),
    ("bm25", 583, 358),
    ("similarity", 617, 378),
    ("verifier", 1041, 469),
):
    vd = jl(f"verdicts_{c}_eval.jsonl")
    check(
        f"contradiction >= 0.5, {c}",
        con,
        sum(v["max_contradiction"] >= 0.5 for v in vd),
    )
    check(f"entailment >= 0.5, {c}", ent, sum(v["max_entailment"] >= 0.5 for v in vd))

# Stages 7-8: flags and results
section("Stages 7-8: flags and results")
check("threshold", 0.95, json.load(open("threshold.json"))["threshold"], tol=0)
expected = {  # flagged, AUROC, AP, P, R, accuracy, statements-only, claim-only, robust AUROC, robust AP
    "gold": (81, 0.715, 0.230, 0.198, 0.500, 0.730, 0.679, 0.775, 0.731, 0.139),
    "bm25": (171, 0.594, 0.136, 0.135, 0.719, 0.477, 0.595, 0.449, 0.584, 0.099),
    "similarity": (181, 0.587, 0.128, 0.133, 0.750, 0.450, 0.565, 0.520, 0.577, 0.092),
    "verifier": (233, 0.599, 0.137, 0.124, 0.906, 0.310, 0.609, 0.501, 0.603, 0.100),
}
for c, (fl, au, ap, p, r, acc, st, cl, rb, rap) in expected.items():
    rows = []
    for x in csv.DictReader(open(f"flags_{c}_eval.csv", encoding="utf-8")):
        rows.append(
            {
                "claim_id": int(x["claim_id"]),
                "wrong": x["wrong"] == "True",
                "flagged": x["flagged"] == "True",
                "score": float(x["score"]),
                "stmt": float(x["stmt_score"]),
                "claim": float(x["claim_score"]),
            }
        )
    rows.sort(key=lambda z: z["claim_id"])
    tp = sum(z["flagged"] and z["wrong"] for z in rows)
    nfl = sum(z["flagged"] for z in rows)
    nw = sum(z["wrong"] for z in rows)
    s = lambda f, rs=rows: [{"wrong": z["wrong"], "score": z[f]} for z in rs]
    check(f"{c}: flagged", fl, nfl)
    check(f"{c}: AUROC", au, auroc(s("score")), tol=0.0015)
    check(f"{c}: AP", ap, average_precision(s("score")), tol=0.0015)
    check(f"{c}: precision at 0.95", p, tp / nfl, tol=0.0015)
    check(f"{c}: recall at 0.95", r, tp / nw, tol=0.0015)
    check(
        f"{c}: accuracy at 0.95",
        acc,
        sum(z["flagged"] == z["wrong"] for z in rows) / len(rows),
        tol=0.0015,
    )
    check(f"{c}: AUROC statements only", st, auroc(s("stmt")), tol=0.0015)
    check(f"{c}: AUROC claim check only", cl, auroc(s("claim")), tol=0.0015)
    rr = [z for z in rows if z["claim_id"] not in DOUBTFUL]
    check(f"{c}: robustness AUROC", rb, auroc(s("score", rr)), tol=0.0015)
    check(f"{c}: robustness AP", rap, average_precision(s("score", rr)), tol=0.0015)
    check(
        f"{c}: missed errors (SQ3 sheet rows)",
        {"gold": 16, "bm25": 9, "similarity": 8, "verifier": 3}[c],
        sum(z["wrong"] and not z["flagged"] for z in rows),
    )

# Stage 9: annotation sheet
section("Stage 9: statement annotation sheet")
ann = list(csv.DictReader(open("statement_annotation.csv", encoding="utf-8-sig")))
check("annotation statements", 82, len(ann))
check("annotation outputs", 15, len({r["output_id"] for r in ann}))

print(
    f"\n{sum(results)} OK, {len(results) - sum(results)} DIFF, of {len(results)} checks"
)
