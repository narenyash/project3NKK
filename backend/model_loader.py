"""
Loads GPT-2, spaCy (via text_utils), both trained models, and the corpus baseline ONCE
at process startup. Nothing here reloads per-request - `get_state()` returns the same
cached objects every call.

Import strategy: reuses the existing Phase 2 scripts/ modules directly (features.py,
text_utils.py, build_sentences.py's junk-fragment filter) via sys.path, rather than
copying/reimplementing them here. The feature-extraction pipeline was validated end-to-end
across Phase 2 and Phase 4 - forking it for the backend would immediately risk drift
between what was evaluated in EVALUATION.md and what the live service actually runs.
"""
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
SCRIPTS_DIR = ROOT / "scripts"
DATASET_DIR = ROOT / "dataset"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import joblib

import features as F  # noqa: E402  (path inserted above)
from text_utils import split_sentences, using_spacy  # noqa: E402
from build_sentences import BARE_FRAGMENT_RE, CITATION_APOSTROPHE_RE  # noqa: E402

ESSAY_MODEL_FEATURES = [
    "perplexity_z", "logprob_variance_z", "sentence_length_variance_z",
    "syntactic_depth_variance_z", "perplexity_variance_z", "ngram_repeat_rate_z",
    "transition_repetition_rate_z",
]
SENTENCE_MODEL_FEATURES = ["perplexity_z", "logprob_variance_z"]

# Fairness/confidence numbers, sourced directly from EVALUATION.md - not placeholders.
FPR_STATS = {
    "essay_model_sentence_level": {"human": 0.1648, "esl": 0.3245},
    "essay_model_essay_level": {"human": 0.2000, "esl": 0.2633},
    "sentence_model": {"human": 0.5635, "esl": 0.7739},
}

_state = None


class AppState:
    """Holds every loaded, ready-to-use component the pipeline needs: GPT-2, both
    trained models, and the corpus baseline. One instance is created (by get_state(),
    below) and reused for the life of the process - constructing it is the slow part
    (loading GPT-2 alone takes a few seconds), so it must happen exactly once at
    startup, never per-request."""

    def __init__(self):
        print("[startup] Loading GPT-2...")
        self.gpt2_model, self.gpt2_tokenizer = F.get_model()

        print("[startup] Confirming spaCy loaded...")
        if not using_spacy():
            raise RuntimeError("spaCy (en_core_web_sm) failed to load - required for sentence "
                                "segmentation and burstiness features.")

        print("[startup] Loading essay-level model...")
        self.essay_model = joblib.load(DATASET_DIR / "logistic_regression_model.joblib")

        print("[startup] Loading sentence-level model...")
        self.sentence_model = joblib.load(DATASET_DIR / "sentence_level_model.joblib")

        print("[startup] Loading corpus baseline...")
        self.baseline = json.loads((DATASET_DIR / "corpus_baseline.json").read_text(encoding="utf-8"))

        print("[startup] All components loaded.")

    # Checked by GET /health (main.py) - true only once every component genuinely
    # finished loading, so a health check during startup correctly reports "not ready"
    # rather than a premature "ok".
    def ready(self) -> bool:
        return all([
            self.gpt2_model is not None,
            self.essay_model is not None,
            self.sentence_model is not None,
            self.baseline is not None,
            using_spacy(),
        ])


# Lazy singleton: the first call constructs AppState (slow - loads GPT-2 and both
# models); every call after that returns the same cached instance instantly. Called
# once eagerly at import time in main.py (so startup cost happens at process boot, not
# on the first request) and again per-request inside pipeline.py's analyze_essay().
def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
