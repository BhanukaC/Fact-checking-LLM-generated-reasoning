import json, collections, random, urllib.request

# 1. Download the AVeriTeC dev set
url = "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json"
urllib.request.urlretrieve(url, "dev.json")
data = json.load(open("dev.json", encoding="utf-8"))
print("claims:", len(data))
print(collections.Counter(x["label"] for x in data))

# 2. Stratified sample with a fixed seed
SEED = 2026
rng = random.Random(SEED)
eval_ids, tune_ids = [], []
for label in ["Supported", "Refuted"]:
    ids = sorted(i for i, x in enumerate(data) if x["label"] == label)
    rng.shuffle(ids)
    eval_ids += ids[:50]
    tune_ids += ids[50:60]
assert not set(eval_ids) & set(tune_ids)


def row(i, split):
    x = data[i]
    return {
        "claim_id": i,  # index in dev.json
        "split": split,
        "claim": x["claim"],
        "claim_date": x["claim_date"],
        "speaker": x.get("speaker"),
        "gold_label": x["label"],
        "gold_justification": x["justification"],
        "gold_questions": x.get("questions", []),
    }


claims = [row(i, "eval") for i in sorted(eval_ids)] + [
    row(i, "tune") for i in sorted(tune_ids)
]
json.dump(
    {"seed": SEED, "source": "AVeriTeC dev.json", "claims": claims},
    open("sampled_claims.json", "w", encoding="utf-8"),
    indent=2,
    ensure_ascii=False,
)

print(collections.Counter((c["split"], c["gold_label"]) for c in claims))
print("first eval claim:", claims[0]["claim_id"], claims[0]["claim"])
