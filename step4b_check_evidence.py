import csv, json, re
from collections import Counter
from datetime import date
from urllib.parse import urlparse

claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
}
text = open("data/dev_top_k_sentences.json", encoding="utf-8").read()
evidence = {
    e["claim_id"]: e for e in (json.loads(l) for l in text.split("\n") if l.strip())
}

FACTCHECK_DOMAINS = {
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "fullfact.org",
    "checkyourfact.com",
    "leadstories.com",
    "factcheck.afp.com",
    "africacheck.org",
    "altnews.in",
    "boomlive.in",
    "factly.in",
    "healthfeedback.org",
    "sciencefeedback.co",
    "truthorfiction.com",
    "newschecker.in",
    "factcheckni.org",
}


def real_url(url):
    # web.archive.org/web/<timestamp>/<original url>  ->  original url
    m = re.match(r"https?://web\.archive\.org/web/\d+[a-z_]*/(.+)", url)
    return m.group(1) if m else url


def domain(url):
    d = urlparse(real_url(url)).netloc.lower()
    return d[4:] if d.startswith("www.") else d


def is_factcheck(url):
    u = real_url(url).lower()
    return domain(url) in FACTCHECK_DOMAINS or any(
        k in u for k in ("fact-check", "factcheck", "fact_check", "fact-checker")
    )


def url_date(url):
    # dates like /2020/11/03/ or /2020/11/ in the path; None if not found
    m = re.search(r"/(20\d\d)/(\d{1,2})(?:/(\d{1,2}))?/", real_url(url))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 28))
    except ValueError:
        return None


def claim_date(c):
    d, m, y = (int(x) for x in c["claim_date"].split("-"))
    return date(y, m, d)


# 1. Every sampled claim has evidence, and the claim text matches
missing = [cid for cid in claims if cid not in evidence]
mismatch = [
    cid
    for cid in claims
    if cid in evidence
    and evidence[cid]["claim"].strip() != claims[cid]["claim"].strip()
]
print(
    "sampled claims:",
    len(claims),
    "| missing evidence:",
    len(missing),
    "| claim text mismatch:",
    len(mismatch),
)

# 2. Count problems per claim
totals = Counter()
domains = Counter()
per_claim = []
for cid, c in sorted(claims.items()):
    sents = evidence[cid]["top_100"]
    seen, kept = set(), 0
    n_dup = n_fc = n_after = n_dated = 0
    for s in sents:
        key = s["sentence"].strip().lower()
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        domains[domain(s["url"])] += 1
        if is_factcheck(s["url"]):
            n_fc += 1
            continue
        d = url_date(s["url"])
        if d:
            n_dated += 1
            if d > claim_date(c):
                n_after += 1
                continue
        kept += 1
    per_claim.append(
        {
            "claim_id": cid,
            "split": c["split"],
            "sentences": len(sents),
            "duplicates": n_dup,
            "factcheck": n_fc,
            "url_dated": n_dated,
            "after_claim_date": n_after,
            "kept": kept,
        }
    )
    totals.update(
        {
            "sentences": len(sents),
            "duplicates": n_dup,
            "factcheck": n_fc,
            "url_dated": n_dated,
            "after_claim_date": n_after,
            "kept": kept,
        }
    )

with open("evidence_per_claim.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(per_claim[0].keys()))
    w.writeheader()
    w.writerows(per_claim)

print("\nTotals over", len(claims), "claims:", dict(totals))
kept = sorted(r["kept"] for r in per_claim)
print(
    "kept sentences per claim: min",
    kept[0],
    "| median",
    kept[len(kept) // 2],
    "| max",
    kept[-1],
)
print("claims with fewer than 20 kept:", sum(k < 20 for k in kept))

print("\nTop 25 source domains:")
for d, n in domains.most_common(25):
    print(f'  {n:5}  {d}{"   <- fact-check" if d in FACTCHECK_DOMAINS else ""}')

# 3. Coverage eyeball: for 3 claims, show an LLM reasoning and the kept sentences sharing the most words with it
outputs = [json.loads(l) for l in open("outputs_eval.jsonl", encoding="utf-8")]
words = lambda t: set(re.findall(r"[a-z0-9]+", t.lower())) - {
    "the",
    "a",
    "of",
    "in",
    "to",
    "and",
    "was",
    "is",
    "that",
    "for",
    "on",
}
for cid in [36, 145, 177]:
    if cid not in evidence:
        continue
    r = next((o for o in outputs if o["claim_id"] == cid), None) or next(
        (
            json.loads(l)
            for l in open("outputs_tune.jsonl", encoding="utf-8")
            if json.loads(l)["claim_id"] == cid
        ),
        None,
    )
    if not r:
        continue
    print("\n" + "=" * 80)
    print(f'CLAIM {cid}: {claims[cid]["claim"]}')
    print(f'{r["model"]} reasoning: {r["llm_reasoning"]}')
    rw = words(r["llm_reasoning"])
    cands = [s for s in evidence[cid]["top_100"] if not is_factcheck(s["url"])]
    cands.sort(key=lambda s: len(rw & words(s["sentence"])), reverse=True)
    print("Most overlapping evidence sentences:")
    for s in cands[:6]:
        print(f'  - {s["sentence"][:200]}  [{domain(s["url"])}]')
