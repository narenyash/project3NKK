"""
Closes DATASET.md's Known Limitation #1: "AI source metadata is undocumented. All 100
`ai` essays come from one unspecified model with no recorded prompt/settings/temperature."

That's true of the ORIGINAL 100 essays in `Class A/` and can't be fixed retroactively -
there's no metadata to recover, and this script does not touch those files. What this
closes is the *process* gap: any essay generated with this script from now on has its
model, exact prompt, generation options, and timestamp written to a companion provenance
log, so this specific undocumented-source problem cannot happen again for new data added
to this corpus.

Output goes to `Class A_v2/` (new files only, never mixed into the original `Class A/`
batch) plus `dataset/ai_essay_generation_log.jsonl` (one record per essay). Folding new
essays into `essays.jsonl`/`splits.json` and retraining on the expanded set is a separate,
deliberate step (see README note at the bottom of this file) - this script only handles
generation + provenance logging.

Follows the same Ollama-as-instrument pattern already established in
`regenerate_hybrids.py`: model name is a module constant, prompt template is inline and
readable, no undocumented magic.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import ollama

from text_utils import word_count

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "Class A_v2"
LOG_PATH = ROOT / "dataset" / "ai_essay_generation_log.jsonl"

MODEL = "mistral:latest"
OPTIONS = {"temperature": 0.9}

# Rotating through several topics addresses part of DATASET.md's OTHER known limitation
# (94% topic concentration) for any essays generated going forward - not a full fix for
# the existing corpus, just not making that problem worse with every new batch.
TOPICS = [
    "overcoming a significant challenge",
    "a community service experience that changed your perspective",
    "an intellectual curiosity that led you to explore a subject deeply",
    "a time you questioned a belief you had long held",
    "an accomplishment you are proud of and what it taught you",
    "an aspect of your identity or background that has shaped you",
    "a topic, idea, or concept that captivates you so much you lose track of time",
]

PROMPT_TEMPLATE = """Write a college admissions essay (600-900 words) responding to this prompt: \
"Describe {topic}."

Write in first person, with a genuine personal voice and specific, concrete details \
(not generic statements). Return ONLY the essay text, no title, no preamble, no commentary."""


# Generates and saves one essay, and returns its full provenance record (everything
# main() below writes to the JSONL log) - the record is built regardless of whether the
# essay is ever folded into the modeling dataset, since capturing provenance at
# generation time (not reconstructed later) is the entire point of this script.
def generate_one(essay_num: int, topic: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(topic=topic)
    timestamp = datetime.now(timezone.utc).isoformat()

    response = ollama.generate(model=MODEL, prompt=prompt, options=OPTIONS, stream=False)
    text = response["response"].strip()

    filename = f"{essay_num}.txt"
    (OUTPUT_DIR / filename).write_text(text, encoding="utf-8")

    return {
        "id": f"ai_v2_{essay_num:03d}",
        "filename": filename,
        "model": MODEL,
        "options": OPTIONS,
        "prompt_topic": topic,
        "full_prompt": prompt,
        "generated_at_utc": timestamp,
        "word_count": word_count(text),
    }


# Entry point: generates `count` essays starting at `start_num`, cycling through TOPICS,
# and appends every successful generation's provenance record to the JSONL log (failed
# generations are printed but not logged, since there's no essay to attach a record to).
def main(count: int, start_num: int = 1):
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_PATH.parent.mkdir(exist_ok=True)

    print(f"Generating {count} AI essays with model={MODEL}, options={OPTIONS}")
    print(f"Output: {OUTPUT_DIR}/  |  Provenance log: {LOG_PATH}\n")

    records = []
    for i in range(count):
        essay_num = start_num + i
        topic = TOPICS[i % len(TOPICS)]
        print(f"  [{essay_num}] topic: {topic!r}...", end=" ", flush=True)
        try:
            record = generate_one(essay_num, topic)
            records.append(record)
            print(f"ok ({record['word_count']} words)")
        except Exception as e:
            print(f"FAILED: {e}")

    with LOG_PATH.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{len(records)}/{count} essays generated and logged.")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)

# --- To actually add generated essays to the modeling dataset (not done by this script):
# 1. Review the output in Class A_v2/ - AI generation can produce essays that are too
#    similar to each other or too formulaic; spot-check before including.
# 2. Extend build_dataset.py to also ingest Class A_v2/ (or merge folders) with the
#    provenance log's data replacing the current
#    "ai-generated (model/settings undocumented)" placeholder `source` string for these
#    new records specifically - the original 100 keep their honest "undocumented" source,
#    this is additive, not a rewrite of history.
# 3. Re-run quality_checks.py, topic_tagger.py, and make_split.py on the expanded set.
# 4. Re-extract features and retrain both models (scripts/features.py + Phase 2 training
#    scripts) - the existing .joblib models and EVALUATION.md numbers were trained/
#    measured on the original 296-essay set and do not reflect the expanded corpus until
#    this step is done.
