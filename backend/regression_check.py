"""
Step 1 regression check: confirm GPT-2, spaCy, both models, and the baseline all load
via model_loader.py, and that the ported code reproduces the exact perplexity value from
Phase 2's original gpt2_sanity_check.py (37.09 for the hardcoded test sentence) - proof
nothing broke in porting the pipeline into backend/.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import get_state
import features as F

SENTENCE = "The cat sat quietly on the windowsill, watching the rain fall outside."
EXPECTED_PERPLEXITY = 37.09  # from scripts/gpt2_sanity_check.py's original output


def main():
    print("=" * 60)
    print("BACKEND STEP 1: REGRESSION CHECK")
    print("=" * 60)

    state = get_state()
    print(f"\nready() = {state.ready()}")
    print(f"essay_model features expected: 7, actual coef count: {len(state.essay_model.coef_[0])}")
    print(f"sentence_model features expected: 2, actual coef count: {len(state.sentence_model.coef_[0])}")
    print(f"baseline features loaded: {list(state.baseline.keys())}")

    token_lps = F.sentence_token_logprobs(SENTENCE, "")
    perplexity = F.sentence_perplexity(token_lps)
    print(f"\nTest sentence: {SENTENCE!r}")
    print(f"Perplexity (ported code): {perplexity:.2f}")
    print(f"Perplexity (original Phase 2 sanity check): {EXPECTED_PERPLEXITY}")
    match = abs(perplexity - EXPECTED_PERPLEXITY) < 0.01
    print(f"Match: {'YES' if match else 'NO - INVESTIGATE'}")


if __name__ == "__main__":
    main()
