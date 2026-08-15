"""
Phase 1, Step 8: download the public ELLIPSE corpus (English Language Learner
essays) and normalize it into a standalone fairness-check set. Kept completely
separate from dataset/essays.jsonl and dataset/splits.json - never merged, never
used for training. Reserved for the Phase 4 false-positive-rate check.
"""
import csv
import io
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "dataset" / "ellipse_fairness.jsonl"

SOURCE_URL = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/ELLIPSE_Final_github_train.csv"


# Entry point: downloads the ELLIPSE CSV over HTTP, normalizes rows into this project's
# record shape, and writes dataset/ellipse_fairness.jsonl - no API key/auth needed, it's
# a public GitHub-hosted file.
def main():
    print("=" * 60)
    print("STEP 8: ELLIPSE FAIRNESS SLICE")
    print("=" * 60)
    print(f"Downloading {SOURCE_URL} ...")

    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(raw))
    records = []
    for row in reader:
        eid = f"ellipse_{row['text_id_kaggle']}"
        records.append({
            "id": eid,
            "text": row["full_text"].strip(),
            "label": "human",
            "is_ell": True,
            "source": "ELLIPSE corpus (scrosseye/ELLIPSE-Corpus, public GitHub)",
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} ELL essays to {OUT_PATH.relative_to(ROOT)}")
    print("This file is NOT part of essays.jsonl/splits.json and is never used for training - "
          "reserved for the Phase 4 false-positive-rate fairness check.")


if __name__ == "__main__":
    main()
