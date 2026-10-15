# step5_decompose.py: split each LLM reasoning into atomic, standalone statements
import argparse, json, os, re, time
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
DECOMPOSER = "openai/gpt-6.1-sol"
PROMPT_VERSION = "d1"
TYPES = {"fact", "meta", "hedge"}

SYSTEM_PROMPT = (
    "You split a fact-checking explanation into atomic statements. Rules: "
    "1. Each statement contains exactly one checkable fact. "
    "2. Each statement must be understandable on its own: replace pronouns and vague references "
    "(he, the seat, this law) with what they refer to, using the claim for context. "
    "3. Do not add, remove, correct or judge information. Keep every number, date, name and quantity exactly as written. "
    "4. Tag each statement: fact = a checkable claim about the world; meta = about the claim, the verdict or the "
    "reasoning itself, or citing fact-checkers or debunking; hedge = vague, opinion or not checkable. "
    "5. Keep the original order."
)
USER_TEMPLATE = (
    "Claim (context only, do not decompose it): {claim}\n"
    "Claim date: {date}\n\n"
    "Explanation to decompose:\n{reasoning}\n\n"
    "Answer with JSON only:\n"
    '{{"statements": [{{"text": "...", "type": "fact" or "meta" or "hedge"}}]}}'
)
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
NUM = re.compile(r"\d[\d,.]*\d|\d")


def readable_date(s):
    d, m, y = (int(x) for x in s.split("-"))  # same date fix as Step 2
    return f"{d} {MONTHS[m - 1]} {y}"


def numbers(text):
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


def call(messages):
    body = {
        "model": DECOMPOSER,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(4):
        try:
            r = requests.post(
                URL, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300
            )
        except requests.exceptions.RequestException as e:
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"network error: {e}")
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("gave up after retries")


def parse(text):
    """Return a list of (text, type) or None if the output is unusable."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    try:
        items = json.loads(text[start : end + 1])["statements"]
    except (ValueError, KeyError, TypeError):
        return None
    out = []
    for s in items:
        t, ty = str(s.get("text", "")).strip(), str(s.get("type", "")).strip().lower()
        if not t or ty not in TYPES:
            return None
        out.append((t, ty))
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "tune"], required=True)
    ap.add_argument("--limit", type=int, help="first N claims, all three models")
    args = ap.parse_args()

    claims = {
        c["claim_id"]: c
        for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
    }
    outputs = [
        json.loads(l) for l in open(f"outputs_{args.split}.jsonl", encoding="utf-8")
    ]
    if args.limit:
        keep = sorted({o["claim_id"] for o in outputs})[: args.limit]
        outputs = [o for o in outputs if o["claim_id"] in keep]

    out_path = f"statements_{args.split}.jsonl"
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            if r["valid"]:
                done.add(r["output_id"])  # failed rows are retried on the next run

    with open(out_path, "a", encoding="utf-8") as f:
        for o in outputs:
            output_id = f"{o['claim_id']}|{o['model']}"
            if output_id in done or not o["llm_reasoning"]:
                continue
            c = claims[o["claim_id"]]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        claim=c["claim"],
                        date=readable_date(c["claim_date"]),
                        reasoning=o["llm_reasoning"],
                    ),
                },
            ]
            attempts, parsed = [], None
            for _ in range(2):  # one retry on unusable output
                try:
                    resp = call(messages)
                except RuntimeError as e:
                    attempts.append({"error": str(e)})
                    break
                choice = (resp.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content")
                attempts.append(
                    {
                        "raw": text,
                        "finish_reason": choice.get("finish_reason"),
                        "provider": resp.get("provider"),
                        "usage": resp.get("usage"),
                    }
                )
                parsed = parse(text)
                if parsed:
                    break

            short = o["model"].split("/")[1].split("-")[0]  # gemini / claude / gpt
            statements = [
                {"sid": f"{o['claim_id']}-{short}-{i}", "text": t, "type": ty}
                for i, (t, ty) in enumerate(parsed or [], start=1)
            ]
            # every number in a statement must come from the reasoning, the claim or its date
            allowed = (
                numbers(o["llm_reasoning"])
                | numbers(c["claim"])
                | numbers(c["claim_date"])
            )
            new_numbers = sorted(
                {n for s in statements for n in numbers(s["text"])} - allowed
            )
            rec = {
                "output_id": output_id,
                "claim_id": o["claim_id"],
                "model": o["model"],
                "decomposer": DECOMPOSER,
                "prompt_version": PROMPT_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "statements": statements,
                "valid": parsed is not None,
                "new_numbers": new_numbers,
                "size_flag": not (1 <= len(statements) <= 15),
                "attempts": attempts,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_fact = sum(s["type"] == "fact" for s in statements)
            flags = []
            if not rec["valid"]:
                flags.append("INVALID")
            if new_numbers:
                flags.append(f"NEW NUMBERS {new_numbers}")
            if rec["size_flag"]:
                flags.append("SIZE")
            print(
                f"{output_id}: {len(statements)} statements ({n_fact} fact) {' '.join(flags)}"
            )


if __name__ == "__main__":
    main()
