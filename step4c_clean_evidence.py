import csv, json, re
from collections import Counter
from datetime import date
from urllib.parse import urlparse

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
    "vishvasnews.com",
    "factcheckni.org",
    "misbar.com",
    "teyit.org",
}
STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "and",
    "was",
    "is",
    "that",
    "for",
    "on",
    "it",
    "by",
    "with",
    "as",
    "at",
    "be",
    "are",
    "this",
}


def real_url(url):
    m = re.match(r"https?://web\.archive\.org/web/\d+[a-z_]*/(.+)", url or "")
    return m.group(1) if m else (url or "")


def norm_url(url):
    # compare URLs without http/https, www, query string or trailing slash
    p = urlparse(real_url(url).lower())
    host = p.netloc[4:] if p.netloc.startswith("www.") else p.netloc
    return host + p.path.rstrip("/")


def domain(url):
    return norm_url(url).split("/")[0]


def is_factcheck(url):
    u = real_url(url).lower()
    return domain(url) in FACTCHECK_DOMAINS or any(
        k in u for k in ("fact-check", "factcheck", "fact_check")
    )


def url_date(url):
    m = re.search(r"/(20\d\d)/(\d{1,2})(?:/(\d{1,2}))?/", real_url(url))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 28))
    except ValueError:
        return None


def parse_date(s):
    d, m, y = (int(x) for x in s.split("-"))
    return date(y, m, d)


def content_words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


# Load the sample, the full dev set (for the claim's source URLs) and the BM25 candidates
claims = {
    c["claim_id"]: c
    for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
}
dev = json.load(open("dev.json", encoding="utf-8"))
raw = open("data/dev_top_k_sentences.json", encoding="utf-8").read()
evidence = {
    e["claim_id"]: e for e in (json.loads(l) for l in raw.split("\n") if l.strip())
}

cleaned, removed_rows = {}, []
reasons = Counter()

for cid, c in sorted(claims.items()):
    d = dev[cid]
    own_urls = {
        norm_url(u)
        for u in (
            d.get("original_claim_url"),
            d.get("cached_original_claim_url"),
            d.get("fact_checking_article"),
        )
        if u
    }
    claim_words = content_words(c["claim"])
    cdate = parse_date(c["claim_date"])

    seen, kept = set(), []
    for rank, s in enumerate(evidence[cid]["top_100"]):
        sent, url = s["sentence"].strip(), s["url"]
        key = sent.lower()
        words = content_words(sent)

        if sent.lower().startswith("pingback") or len(sent.split()) < 5:
            reason = "noise"
        elif key in seen:
            reason = "duplicate"
        elif is_factcheck(url):
            reason = "factcheck_source"
        elif norm_url(url) in own_urls:
            reason = "claim_own_source"
        elif claim_words and len(claim_words & words) / len(claim_words) >= 0.8:
            reason = "copy_of_claim"
        elif url_date(url) and url_date(url) > cdate:
            reason = "after_claim_date"
        else:
            reason = None

        seen.add(key)
        if reason:
            reasons[reason] += 1
            removed_rows.append(
                {"claim_id": cid, "reason": reason, "sentence": sent, "url": url}
            )
        else:
            kept.append(
                {
                    "sid": f"{cid}-{rank}",
                    "sentence": sent,
                    "url": url,
                    "domain": domain(url),
                }
            )

    cleaned[cid] = {
        "claim": c["claim"],
        "claim_date": c["claim_date"],
        "split": c["split"],
        "sentences": kept,
    }

json.dump(
    cleaned,
    open("evidence_clean.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
with open("evidence_removed.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["claim_id", "reason", "sentence", "url"])
    w.writeheader()
    w.writerows(removed_rows)

counts = sorted(len(v["sentences"]) for v in cleaned.values())
print("removed by reason:", dict(reasons))
print(
    "kept sentences:",
    sum(counts),
    "| per claim: min",
    counts[0],
    "| median",
    counts[len(counts) // 2],
    "| max",
    counts[-1],
)

# Spot check: what was removed for claim 145, and whether the Pence quote survived
print("\nClaim 145 removed:")
for r in removed_rows:
    if r["claim_id"] == 145:
        print(f"  [{r['reason']}] {r['sentence'][:120]}")
pence = [
    s for s in cleaned[145]["sentences"] if "hesitated for a month" in s["sentence"]
]
print(
    "\nPence quote still in claim 145 evidence:",
    bool(pence),
    pence[0]["url"] if pence else "",
)
