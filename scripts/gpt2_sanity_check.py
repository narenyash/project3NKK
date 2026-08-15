"""
Phase 2, Step 0 (blocking sanity check): confirm GPT-2 loads locally and we can extract
raw per-token log-probabilities from its logits - as a measurement instrument only, never
as a judge that outputs a verdict directly.
"""
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Windows console defaults to cp1252, which can't render GPT-2's byte-level BPE
# markers (e.g. 'Ġ' for a leading space) - force UTF-8 stdout so printing tokens works.
sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "gpt2"  # base GPT-2, ~124M params

SENTENCE = "The cat sat quietly on the windowsill, watching the rain fall outside."


def main():
    print("=" * 60)
    print("PHASE 2 STEP 0: GPT-2 SANITY CHECK")
    print("=" * 60)
    print(f"Loading tokenizer and model ({MODEL_NAME})...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"\nSentence: {SENTENCE!r}\n")

    inputs = tokenizer(SENTENCE, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    # log P(token_i | token_<i) for i = 1..n-1: logits[i-1] predicts token i
    log_probs_full = torch.log_softmax(logits, dim=-1)  # (1, seq_len, vocab)
    token_ids = input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    per_token_logprobs = []
    for i in range(1, len(token_ids)):
        predicted_logits_position = i - 1
        actual_token_id = token_ids[i].item()
        lp = log_probs_full[0, predicted_logits_position, actual_token_id].item()
        per_token_logprobs.append(lp)

    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"\nPer-token log-probs (for tokens 2..{len(tokens)}, first token has no "
          f"preceding context so no log-prob):")
    for tok, lp in zip(tokens[1:], per_token_logprobs):
        print(f"  {tok!r:20s} log P = {lp:.4f}")

    avg_neg_logprob = -sum(per_token_logprobs) / len(per_token_logprobs)
    perplexity = torch.exp(torch.tensor(avg_neg_logprob)).item()
    print(f"\nAverage negative log-likelihood: {avg_neg_logprob:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("\nSANITY CHECK PASSED: forward pass and per-token log-prob extraction both work.")


if __name__ == "__main__":
    main()
