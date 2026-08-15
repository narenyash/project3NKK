"""Shared text helpers: sentence splitting (spaCy, regex fallback) and word counting."""
import re

_NLP = None
_NLP_LOADED = False


def _get_nlp():
    """Loads spaCy's en_core_web_sm once and caches it (module-level singleton, same
    pattern as features.py's get_model()). Returns None (never raises) if spaCy or the
    model isn't installed, so callers can fall back to the regex splitter below instead
    of crashing - using_spacy() exposes that same None-check to other modules."""
    global _NLP, _NLP_LOADED
    if not _NLP_LOADED:
        _NLP_LOADED = True
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
        except Exception:
            _NLP = None
    return _NLP


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Leftover source-file header from the original Tab_N.txt / Tab_N_hybrid.txt split
# (Phase 1) - e.g. "Tab 50\n" - not essay content, but fused onto the front of the real
# first sentence with no punctuation between them (spaCy can't split on a bare newline),
# so GPT-2 scores the whole fused blob including the header as one "sentence." A bare
# document label like "Tab 50" is wildly improbable as sentence text (observed up to 80x
# local perplexity), silently distorting sentence 0's perplexity for ~half the corpus.
# Anchored to match ONLY a line that is exactly "Tab" + digits and nothing else - this
# does NOT match cases where the essay's own (possibly AI-rewritten) content legitimately
# starts with something like "Tab 18: Unleashing Creativity through Code" (real title,
# not the raw artifact), since that has trailing text after the number.
_TAB_HEADER_RE = re.compile(r"^Tab \d+[ \t]*\n")


def split_sentences(text: str) -> list[str]:
    """The one sentence-segmentation function used everywhere in this project (Phase 1
    dataset building, Phase 2 feature extraction, and the live backend pipeline) - never
    reimplemented elsewhere, so segmentation behavior can't drift between training and
    inference. Strips the Tab-header artifact once at the start, then uses spaCy if
    available (see _get_nlp()) or falls back to a plain punctuation-based regex split."""
    text = (text or "").strip()
    if not text:
        return []
    text = _TAB_HEADER_RE.sub("", text, count=1)
    nlp = _get_nlp()
    if nlp is not None:
        # spaCy's sentence segmenter can mis-split a contraction into its own "sentence"
        # when it uses a curly apostrophe (U+2019) instead of a straight one - discovered
        # via a real mis-split found in Phase 2 (Step 4): "There's no denying its wow
        # factor." got broken into "There's" + "no denying its wow factor." Segmentation
        # boundaries are computed on a straight-apostrophe-normalized copy of the text,
        # but the returned sentences are sliced from the ORIGINAL string by character
        # offset (a 1-char-for-1-char swap preserves alignment), so the original curly
        # apostrophes/quotes are preserved in the output - only the segmentation decision
        # changes, not the text content.
        normalized = text.replace("’", "'")
        doc = nlp(normalized)
        sentences = []
        for s in doc.sents:
            original_slice = text[s.start_char:s.end_char].strip()
            if original_slice:
                sentences.append(original_slice)
        return sentences
    # Regex fallback (used if spaCy/model isn't available)
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    """Whitespace-split word count - the same simple definition MIN_WORDS/MAX_WORDS are
    checked against in backend/pipeline.py, and mirrored client-side in
    frontend/src/components/EssayInput.jsx's wordCount()."""
    return len((text or "").split())


def using_spacy() -> bool:
    """True if spaCy + en_core_web_sm loaded successfully - backend/model_loader.py's
    AppState.ready() calls this to confirm the backend is fully up, not just GPT-2."""
    return _get_nlp() is not None
