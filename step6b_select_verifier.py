# step6b_select_verifier.py: verifier-guided evidence selection (RAV-style, training-free)
# For each item: similarity top 20 from the claim's pool, scored by the NLI verifier,
# keep the 5 with the highest max(entailment, contradiction).
import argparse, json, time
import torch
from sentence_transformers import SentenceTransformer
from step6_select_evidence import DENSE_MODEL, load_items, record
from step7_verify import load_model, score

K, CANDIDATES = 5, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "tune"], required=True)
    args = ap.parse_args()

    claims = [
        c
        for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
        if c["split"] == args.split
    ]
    pool = json.load(open("evidence_clean.json", encoding="utf-8"))
    items = load_items(args.split, claims)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = SentenceTransformer(DENSE_MODEL, device=device)
    tok, model, idx = load_model(device)

    # 1. Similarity top 20 per item (same pool and query as Step 6)
    cands = []  # (item_id, claim_id, query, [20 sentences])
    for c in claims:
        cid = c["claim_id"]
        sents = pool[str(cid)]["sentences"]
        texts = [s["sentence"] for s in sents]
        p_emb = encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        q_emb = encoder.encode(
            [q for _, q in items[cid]],
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        sims = (q_emb @ p_emb.T).cpu()
        for qi, (item_id, query) in enumerate(items[cid]):
            top = torch.topk(sims[qi], CANDIDATES).indices.tolist()
            cands.append((item_id, cid, query, [sents[i] for i in top]))

    # 2. NLI on every unique (passage, query) pair, shortest first
    pairs = sorted(
        {(s["sentence"], q) for _, _, q, ss in cands for s in ss},
        key=lambda x: len(x[0]) + len(x[1]),
    )
    t0 = time.time()
    probs = dict(zip(pairs, score(pairs, tok, model, idx, device)))
    print(f"{len(pairs)} pairs scored in {time.time() - t0:.0f}s")

    # 3. Keep the 5 passages the verifier is most confident about
    def conf(s, q):
        p = probs[(s["sentence"], q)]
        return max(p["entailment"], p["contradiction"])

    overlap = []
    with open(
        f"evidence_selected_verifier_{args.split}.jsonl", "w", encoding="utf-8"
    ) as f:
        for item_id, cid, query, ss in cands:
            ranked = sorted(ss, key=lambda s: -conf(s, query))[:K]
            f.write(
                record(
                    item_id,
                    cid,
                    "verifier",
                    query,
                    [
                        {
                            "eid": s["sid"],
                            "text": s["sentence"],
                            "score": round(conf(s, query), 4),
                        }
                        for s in ranked
                    ],
                )
                + "\n"
            )
            overlap.append(len({s["sid"] for s in ranked} & {s["sid"] for s in ss[:K]}))
    print(
        f"{len(cands)} items; on average {sum(overlap) / len(overlap):.1f} of the 5 passages "
        f"are also in the similarity top 5"
    )


if __name__ == "__main__":
    main()
