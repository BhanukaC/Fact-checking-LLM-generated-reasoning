import argparse, json, os, time
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"

# Replace with the exact IDs from openrouter.ai/models
MODELS = [
    "google/gemini-3.8-flash",
    "anthropic/claude-sonnet-5",
    "openai/gpt-6.1-sol",
]

# Third-party hosts (Claude Platform on AWS, Azure) applied their own content filters
# to Claude and blocked most claims, so we route Claude only through Anthropic's API.
PROVIDER_PREFS = {
}

PROMPT_VERSION = "v2"
SYSTEM_PROMPT = (
    "You are a fact-checker. You will be given a claim, the date it was made and who made it. "
    "Decide whether the claim is SUPPORTED or REFUTED using only your own knowledge. "
    "Reason as of the claim date: do not rely on events that happened after it. "
    "Do not refer to fact-checking articles or published verdicts about this claim. "
    "Write your reasoning as short factual sentences, then give the label."
)
USER_TEMPLATE = (
    "Claim: {claim}\n"
    "Claim date: {date}\n"
    "Speaker: {speaker}\n\n"
    "Answer with JSON only, in this exact form:\n"
    '{{"reasoning": "<your reasoning>", "label": "SUPPORTED" or "REFUTED"}}'
)


def build_messages(c):
    user = USER_TEMPLATE.format(
        claim=c["claim"], date=c["claim_date"], speaker=c["speaker"] or "Unknown"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call(model, messages):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 4000}
    if model in PROVIDER_PREFS:
        body["provider"] = PROVIDER_PREFS[model]
    for attempt in range(4):
        try:
            r = requests.post(
                URL, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300
            )
        except (
            requests.exceptions.RequestException
        ) as e:  # timeout or connection problem
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"network error: {e}")
        if r.status_code == 200:
            return r.json()
        if r.status_code in (
            429,
            500,
            502,
            503,
        ):  # busy or rate-limited: wait and retry
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("gave up after retries")


def parse(text):
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    label = str(obj.get("label", "")).strip().upper()
    if label not in ("SUPPORTED", "REFUTED") or not obj.get("reasoning"):
        return None
    return obj["reasoning"], label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "tune"], required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    claims = [
        c
        for c in json.load(open("sampled_claims.json", encoding="utf-8"))["claims"]
        if c["split"] == args.split
    ][: args.limit]
    out_path = f"outputs_{args.split}.jsonl"

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["claim_id"], r["model"]))

    with open(out_path, "a", encoding="utf-8") as f:
        for model in MODELS:
            for c in claims:
                if (c["claim_id"], model) in done:
                    continue
                messages = build_messages(c)
                attempts, parsed = [], None
                for _ in range(2):  # one retry if the output is invalid
                    try:
                        resp = call(model, messages)
                    except RuntimeError as e:
                        attempts.append({"error": str(e)})
                        break
                    choice = (resp.get("choices") or [{}])[0]
                    text = (choice.get("message") or {}).get("content")
                    attempts.append({'raw': text,
                                     'finish_reason': choice.get('finish_reason'),
                                     'error': resp.get('error') or choice.get('error'),
                                     'model_returned': resp.get('model'),
                                     'provider': resp.get('provider'), 'usage': resp.get('usage')})
                    parsed = parse(text)
                    if parsed:
                        break
                rec = {
                    "claim_id": c["claim_id"],
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "temperature": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "messages": messages,
                    "llm_reasoning": parsed[0] if parsed else None,
                    "llm_label": parsed[1] if parsed else None,
                    "valid": parsed is not None,
                    "attempts": attempts,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(model, c["claim_id"], rec["llm_label"] or "INVALID")


if __name__ == "__main__":
    main()
