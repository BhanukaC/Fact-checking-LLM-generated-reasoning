# step6_select_evidence.py: pick the evidence passages per item under each condition
# Items: every fact statement (item_id like "36-gemini-2") and every claim ("claim:36").
import argparse, json, re
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

K = 5
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return TOKEN.findall(text.lower())


def gold_passages(d, cid):
    """AVeriTeC gold evidence as passages: 'question answer', skipping unanswerable ones."""
    out = []
    for qi, q in enumerate(d.get("questions", []), start=1):
        for ai, a in enumerate(q.get("answers", []), start=1):
            if a.get("answer_type") == "Unanswerable":
                continue
            ans = (a.get("answer") or "").strip()
            if a.get("answer_type") == "Boolean" and a.get("boolean_explanation"):
                ans = f"{ans}. {a['boolean_explanation'].strip()}"
            if ans:
                out.append(
                    {
                        "eid": f"{cid}-g{qi}.{ai}",
                        "text": f"{q['question'].strip()} {ans}",
                    }
                )
    return out


def load_items(split, claims):
    """Return {claim_id: [(item_id, query_text), ...]}, the claim itself first."""
    items = {c["claim_id"]: [(f"claim:{c['claim_id']}", c["claim"])] for c in claims}
    latest = {}
    for line in open(f"statements_{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["valid"]:
            latest[r["output_id"]] = r  # keep the last valid row per output
    for r in latest.values():
        for s in r["statements"]:
            if s["type"] == "fact":  # only fact statements go to the verifier
                items[r["claim_id"]].append((s["sid"], s["text"]))
    return items


def record(item_id, cid, cond, query, passages):
    return json.dumps(
        {
            "item_id": item_id,
            "claim_id": cid,
            "condition": cond,
            "query": query,
            "passages": passages,
        },
        ensure_ascii=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "tune"], required=True)
    args = ap.parse_args()

    claims = [
        c
        for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
        if c["split"] == args.split
    ]
    dev = json.load(open("dev.json", encoding="utf-8"))
    pool = json.load(open("evidence_clean.json", encoding="utf-8"))  # keys are strings
    items = load_items(args.split, claims)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = SentenceTransformer(DENSE_MODEL, device=device)

    files = {
        cond: open(
            f"evidence_selected_{cond}_{args.split}.jsonl", "w", encoding="utf-8"
        )
        for cond in ("gold", "bm25", "similarity")
    }
    n_items, gold_sizes = 0, []

    for c in claims:
        cid = c["claim_id"]
        sents = pool[str(cid)]["sentences"]
        texts = [s["sentence"] for s in sents]
        queries = items[cid]

        bm25 = BM25Okapi([tokens(t) for t in texts])
        p_emb = encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        q_emb = encoder.encode(
            [q for _, q in queries], normalize_embeddings=True, convert_to_tensor=True
        )
        sims = (q_emb @ p_emb.T).cpu()  # cosine similarity, queries x passages

        gold = gold_passages(dev[cid], cid)
        gold_sizes.append(len(gold))

        for qi, (item_id, query) in enumerate(queries):
            n_items += 1
            files["gold"].write(record(item_id, cid, "gold", query, gold) + "\n")

            # BM25 with the statement as query; ties keep the original (claim-based) order
            scores = bm25.get_scores(tokens(query))
            top = sorted(range(len(texts)), key=lambda i: -scores[i])[:K]
            files["bm25"].write(
                record(
                    item_id,
                    cid,
                    "bm25",
                    query,
                    [
                        {
                            "eid": sents[i]["sid"],
                            "text": texts[i],
                            "score": round(float(scores[i]), 4),
                        }
                        for i in top
                    ],
                )
                + "\n"
            )

            vals, idx = torch.topk(sims[qi], K)
            files["similarity"].write(
                record(
                    item_id,
                    cid,
                    "similarity",
                    query,
                    [
                        {
                            "eid": sents[i]["sid"],
                            "text": texts[i],
                            "score": round(float(v), 4),
                        }
                        for v, i in zip(vals.tolist(), idx.tolist())
                    ],
                )
                + "\n"
            )

    for f in files.values():
        f.close()
    print(
        f"{args.split}: {len(claims)} claims, {n_items} items "
        f"({n_items - len(claims)} fact statements + {len(claims)} claims)"
    )
    print(
        f"gold passages per claim: min {min(gold_sizes)}, mean {sum(gold_sizes)/len(gold_sizes):.1f}, "
        f"max {max(gold_sizes)}, claims with none {sum(g == 0 for g in gold_sizes)}"
    )
    print("dense model:", DENSE_MODEL, "| device:", device)


if __name__ == "__main__":
    main()
