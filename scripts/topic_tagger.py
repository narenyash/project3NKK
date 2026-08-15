"""
Phase 1, Step 3 (fixed for limitation #3): infer a Common-App-style topic per essay
using 3-model consensus across the locally available Ollama models. A single model
previously claimed high confidence on all 300 essays while defaulting 81% of them to
"intellectual_curiosity" - an unreliable signal. Majority vote across independent
models gives an honest confidence signal: 3-way disagreement is now what gets flagged
for manual review, instead of relying on one model's self-reported confidence.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "essays.jsonl"
TOPIC_COVERAGE_PATH = ROOT / "dataset" / "topic_coverage.csv"
LOW_CONFIDENCE_REPORT_PATH = ROOT / "dataset" / "topic_low_confidence.md"

MODELS = ["mistral:latest", "llama3:latest", "gemma4:latest"]

TOPICS = [
    "identity_background",
    "overcoming_challenge",
    "belief_questioned",
    "accomplishment_gratitude",
    "intellectual_curiosity",
    "community_service",
    "topic_that_captivates",
    "other",
]

PROMPT_TEMPLATE = """You are classifying a college admissions essay into exactly one topic category.

Categories:
- identity_background: essay is about the writer's identity, culture, family, or background
- overcoming_challenge: essay centers on overcoming a specific challenge or obstacle
- belief_questioned: essay is about a belief or idea the writer questioned or that was challenged
- accomplishment_gratitude: essay is about an accomplishment and the gratitude/growth that came from it
- intellectual_curiosity: essay is about a topic, idea, or subject that intellectually engages the writer
- community_service: essay centers on service to, or role within, a community
- topic_that_captivates: essay is a free-form topic that captivates the writer, not fitting other categories cleanly
- other: none of the above fit well

Only choose intellectual_curiosity if the essay is specifically about an idea, subject, or field of study \
that fascinates the writer - not just because the writer sounds thoughtful or reflective in general.

Essay:
---
{text}
---

Respond with ONLY a JSON object, no other text, in exactly this form:
{{"topic": "<one of: {topics}>"}}"""


# Filters MODELS (the 3-model consensus list) down to whichever are actually pulled in
# the local Ollama installation - raises rather than silently tagging with fewer models
# than intended if none are available.
def available_models():
    names = {m.model for m in ollama.list().models}
    usable = [m for m in MODELS if m in names]
    if not usable:
        raise RuntimeError(f"None of {MODELS} are available. Run 'ollama pull <model>' first.")
    return usable


# One model's topic guess for one essay - truncates to a 4000-char excerpt (long essays
# don't need the full text for a topic guess, and this keeps prompts fast), parses the
# model's JSON response defensively, and falls back to "other" on any parse/API failure
# rather than crashing the whole tagging run over one bad response.
def infer_topic(model_name: str, text: str):
    excerpt = text[:4000]
    prompt = PROMPT_TEMPLATE.format(text=excerpt, topics=", ".join(TOPICS))
    try:
        response = ollama.generate(model=model_name, prompt=prompt, stream=False)
        raw = response["response"].strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return "other"
        parsed = json.loads(match.group(0))
        topic = parsed.get("topic", "other")
        return topic if topic in TOPICS else "other"
    except Exception as e:
        print(f"    ! Ollama error ({model_name}): {e}")
        return "other"


# Entry point: tags every essay with all available models (model-outer loop - see the
# printed rationale below for why, not essay-outer), takes a majority vote per essay
# (2-or-3 agreement = high confidence, all-disagree = low confidence + all 3 candidates
# recorded), and rewrites essays.jsonl with topic/topic_confidence/topic_candidates.
def main():
    records = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    models = available_models()

    print("=" * 60, flush=True)
    print("STEP 3: TOPIC TAGGING (3-model consensus)", flush=True)
    print("=" * 60, flush=True)
    print(f"Using models: {models}", flush=True)
    print(f"Tagging {len(records)} essays (~{len(records) * len(models)} Ollama calls)...", flush=True)
    print("Looping model-outer (each model loads once and stays resident for all 300 essays, "
          "then the next model loads) instead of essay-outer, to avoid repeatedly swapping "
          "multi-GB models in and out of memory.\n", flush=True)

    votes_by_id = {r["id"]: {} for r in records}
    for model in models:
        print(f"--- Loading {model} and tagging all {len(records)} essays ---", flush=True)
        for i, r in enumerate(records, 1):
            topic = infer_topic(model, r["text"])
            votes_by_id[r["id"]][model] = topic
            print(f"  [{model}] [{i}/{len(records)}] {r['id']} -> {topic}", flush=True)

    disagreements = []
    for r in records:
        votes = votes_by_id[r["id"]]
        counts = Counter(votes.values())
        top_topic, top_count = counts.most_common(1)[0]
        if top_count >= 2:
            topic, confidence = top_topic, "high"
        else:
            topic, confidence = next(iter(votes.values())), "low"
        r["topic"] = topic
        r["topic_confidence"] = confidence
        r["topic_candidates"] = votes
        if confidence == "low":
            disagreements.append((r["id"], r["label"], votes))

    print(f"\nConsensus complete. {len(disagreements)}/{len(records)} essays had 3-way disagreement.", flush=True)

    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    coverage = {t: {"human": 0, "ai": 0, "hybrid": 0} for t in TOPICS}
    for r in records:
        coverage[r["topic"]][r["label"]] += 1

    with TOPIC_COVERAGE_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "human", "ai", "hybrid", "total"])
        for t in TOPICS:
            row = coverage[t]
            writer.writerow([t, row["human"], row["ai"], row["hybrid"], sum(row.values())])

    with LOW_CONFIDENCE_REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Topics needing manual review (3-model disagreement)\n\n")
        if disagreements:
            f.write(f"{len(disagreements)} essay(s) had 3-way model disagreement (no 2/3 majority):\n\n")
            f.write("| id | label | mistral | llama3 | gemma4 |\n|---|---|---|---|---|\n")
            for eid, label, votes in disagreements:
                f.write(
                    f"| {eid} | {label} | {votes.get('mistral:latest','-')} | "
                    f"{votes.get('llama3:latest','-')} | {votes.get('gemma4:latest','-')} |\n"
                )
        else:
            f.write("No 3-way disagreements - every essay had a 2/3+ model majority.\n")

    dominant_topic, dominant_count = Counter(r["topic"] for r in records).most_common(1)[0]
    print(f"\nWrote topic coverage table to {TOPIC_COVERAGE_PATH.relative_to(ROOT)}")
    print(f"Wrote disagreement review list to {LOW_CONFIDENCE_REPORT_PATH.relative_to(ROOT)} "
          f"({len(disagreements)} flagged)")
    print(f"Most common topic: {dominant_topic} ({dominant_count}/{len(records)} = "
          f"{dominant_count/len(records):.0%})")


if __name__ == "__main__":
    main()
