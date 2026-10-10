import json, os, urllib.request
from collections import Counter
from urllib.parse import urlparse

URL = "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/dev_top_k_sentences.json"
PATH = "data/dev_top_k_sentences.json"

# 1. Download once
os.makedirs("data", exist_ok=True)
if not os.path.exists(PATH):
    print("downloading...")
    urllib.request.urlretrieve(URL, PATH)
print("file size (MB):", round(os.path.getsize(PATH) / 1e6, 1))

# 2. Load: the file may be one JSON value or one JSON object per line
text = open(PATH, encoding="utf-8").read()
if text.startswith("version https://git-lfs"):
    raise SystemExit("Got an LFS pointer, not the data. Check the URL uses /resolve/.")
try:
    data = json.loads(text)
    fmt = "single JSON"
except json.JSONDecodeError:
    data = [json.loads(l) for l in text.split('\n') if l.strip()]
    fmt = "JSON lines"
print("format:", fmt, "| top-level type:", type(data).__name__, "| entries:", len(data))

# 3. Show the structure of the first entry
first = data[0] if isinstance(data, list) else data[next(iter(data))]
print(
    "\nkeys in one entry:",
    list(first.keys()) if isinstance(first, dict) else type(first).__name__,
)
print("\nfirst entry (shortened):")
print(json.dumps(first, ensure_ascii=False, indent=2)[:2500])
