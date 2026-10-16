# step7_verify.py: NLI verifier, entailment/neutral/contradiction probabilities per (passage, statement)
# Writes verdicts_<condition>_<split>.jsonl. Labels (supported/refuted/unverified) are added later,
# once the threshold has been chosen on the tune split.
import argparse, json, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
BATCH = 16
MAX_LEN = 512

# Pairs with a known answer, to catch a swapped label order before anything runs
SELF_TEST = [
    (
        "President Donald Trump has not needed supplemental oxygen since being hospitalized, his doctors said.",
        "Trump is not on supplemental oxygen.",
        "entailment",
    ),
    (
        "Trump was given supplemental oxygen at the White House on Friday before going to the hospital.",
        "Trump never received supplemental oxygen.",
        "contradiction",
    ),
    (
        "Kayla Mueller was an American aid worker from Prescott, Arizona.",
        "The Delta Force raid took place in July 2014.",
        "neutral",
    ),
]


def load_model(device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    names = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    idx = {lab: names[lab] for lab in ("entailment", "neutral", "contradiction")}
    return tok, model, idx


@torch.no_grad()
def score(pairs, tok, model, idx, device):
    """pairs: list of (premise, hypothesis). Returns a list of {entailment, neutral, contradiction}."""
    out = []
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i : i + BATCH]
        enc = tok(
            [p for p, _ in batch],
            [h for _, h in batch],
            truncation="only_first",
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        ).to(device)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
        for row in probs:
            out.append({lab: round(float(row[j]), 4) for lab, j in idx.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "tune"], required=True)
    ap.add_argument("--conditions", nargs="+", default=["gold", "bm25", "similarity"])
    ap.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    args = ap.parse_args()

    tok, model, idx = load_model(args.device)
    print("model:", MODEL, "| device:", args.device, "| label ids:", idx)

    # 1. Label-order guard
    test = score([(p, h) for p, h, _ in SELF_TEST], tok, model, idx, args.device)
    for (p, h, want), probs in zip(SELF_TEST, test):
        got = max(probs, key=probs.get)
        print(f"  self-test want {want:13} got {got:13} {probs}")
        if got != want:
            sys.exit("Self-test failed: check the label order before using any output.")

    cache = {}  # (passage, statement) -> probabilities
    for cond in args.conditions:
        items = [
            json.loads(l)
            for l in open(
                f"evidence_selected_{cond}_{args.split}.jsonl", encoding="utf-8"
            )
        ]

        # 2. Score every new pair once, shortest first (less padding, faster)
        todo = sorted(
            {(p["text"], it["query"]) for it in items for p in it["passages"]}
            - cache.keys(),
            key=lambda x: len(x[0]) + len(x[1]),
        )
        t0 = time.time()
        for pair, probs in zip(todo, score(todo, tok, model, idx, args.device)):
            cache[pair] = probs
        print(
            f"{cond}: {len(items)} items, {len(todo)} new pairs scored in {time.time() - t0:.0f}s"
        )

        # 3. Write per-passage probabilities and the per-item maxima
        with open(f"verdicts_{cond}_{args.split}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                ps = [
                    {"eid": p["eid"], **cache[(p["text"], it["query"])]}
                    for p in it["passages"]
                ]
                best_e = max(ps, key=lambda x: x["entailment"])
                best_c = max(ps, key=lambda x: x["contradiction"])
                f.write(
                    json.dumps(
                        {
                            "item_id": it["item_id"],
                            "claim_id": it["claim_id"],
                            "condition": cond,
                            "query": it["query"],
                            "passages": ps,
                            "max_entailment": best_e["entailment"],
                            "best_entailment_eid": best_e["eid"],
                            "max_contradiction": best_c["contradiction"],
                            "best_contradiction_eid": best_c["eid"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Preview only (threshold 0.5); the real threshold is chosen on tune in the next step
        ref = sum(
            1
            for it in items
            if max(
                cache[(p["text"], it["query"])]["contradiction"] for p in it["passages"]
            )
            >= 0.5
        )
        sup = sum(
            1
            for it in items
            if max(
                cache[(p["text"], it["query"])]["entailment"] for p in it["passages"]
            )
            >= 0.5
        )
        print(
            f"   preview at 0.5: max contradiction >= 0.5 for {ref}, max entailment >= 0.5 for {sup} of {len(items)} items"
        )


if __name__ == "__main__":
    main()
