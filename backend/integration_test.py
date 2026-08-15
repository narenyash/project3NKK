"""
Step 5: integration test against the *live*, running /analyze endpoint (backend/main.py
must already be started separately - this script only sends real HTTP requests to it,
it does not import or call analyze_essay() directly like backend/test_pipeline_standalone.py
does). Picks three known essays straight out of dataset/essays.jsonl (one confident
human, one confident AI, one hybrid with known ground-truth sentence_labels) and prints
response time plus a sentence-by-sentence comparison against the hybrid's known labels -
a quick, human-read sanity check that the deployed API is behaving as expected, not an
automated pass/fail test suite.
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8000"


# Looks up one essay's full text + metadata (including sentence_labels for hybrids) by
# its essays.jsonl id, e.g. "h_006", "ai_001", "hy_012".
def load_text(essay_id):
    essays = [json.loads(l) for l in (ROOT / "dataset" / "essays.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {e["id"]: e for e in essays}
    return by_id[essay_id]["text"], by_id[essay_id]


# POSTs to the live /analyze endpoint and times the round-trip - the elapsed time here
# is what api.js's estimateWaitSeconds() constants were calibrated from.
def analyze(text):
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/analyze", json={"text": text})
    elapsed = time.time() - t0
    resp.raise_for_status()
    return resp.json(), elapsed


def main():
    print("=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)
    print(requests.get(f"{BASE_URL}/health").json())

    print("\n" + "=" * 70)
    print("TEST 1: h_006 (confident human, ~800 words)")
    print("=" * 70)
    text, essay = load_text("h_006")
    wc = len(text.split())
    result, elapsed = analyze(text)
    print(f"Word count: {wc}, response time: {elapsed:.2f}s")
    print(f"Essay-level AI-likelihood: {result['essay_level']['ai_likelihood_score']}")
    print(f"Sentence count: {result['sentence_count']}")

    print("\n" + "=" * 70)
    print("TEST 2: ai_001 (confident AI)")
    print("=" * 70)
    text2, essay2 = load_text("ai_001")
    wc2 = len(text2.split())
    result2, elapsed2 = analyze(text2)
    print(f"Word count: {wc2}, response time: {elapsed2:.2f}s")
    print(f"Essay-level AI-likelihood: {result2['essay_level']['ai_likelihood_score']}")
    print(f"Sentence count: {result2['sentence_count']}")

    print("\n" + "=" * 70)
    print("TEST 3: hy_012 (TEST-SPLIT hybrid essay, known sentence_labels)")
    print("=" * 70)
    text3, essay3 = load_text("hy_012")
    wc3 = len(text3.split())
    result3, elapsed3 = analyze(text3)
    print(f"Word count: {wc3}, response time: {elapsed3:.2f}s")
    print(f"Essay-level AI-likelihood: {result3['essay_level']['ai_likelihood_score']}")
    print(f"Sentence count (API): {result3['sentence_count']}, "
          f"known sentence_labels count: {len(essay3['sentence_labels'])}")

    known_labels = essay3["sentence_labels"]
    print(f"\nSentence-by-sentence comparison (API sentence-level flag vs. known label):")
    print(f"{'idx':>4s} {'known_label':>12s} {'sentence_score':>15s} {'flagged?':>10s}  text")
    n = min(len(known_labels), len(result3["sentences"]))
    matches = 0
    for i in range(n):
        api_s = result3["sentences"][i]
        known = known_labels[i]
        score = api_s["sentence_level_score"]
        flagged = score is not None and score >= 0.5
        expected_flag = known == "ai_edited"
        agree = flagged == expected_flag
        matches += agree
        print(f"{i:4d} {known:>12s} {score:15.4f} {str(flagged):>10s} {'OK' if agree else 'X'}  {api_s['text'][:60]!r}")
    print(f"\nAgreement with known labels: {matches}/{n} ({matches/n:.1%})")


if __name__ == "__main__":
    main()
