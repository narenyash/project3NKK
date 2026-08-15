# Legacy raw-data setup scripts

These are the one-off scripts and raw text dumps used **before Phase 1 began**, to turn
essay text into the `Class A/`, `class H/`, `class AH/` folders that `scripts/build_dataset.py`
later consumed. They are not part of the maintained pipeline (that's everything under
`scripts/`, `backend/`, `frontend/`) — kept here for audit/history only, per
`PHASE1_PROCESS.md` §0 ("present from earlier work, not part of this pipeline, kept as-is").

None of these scripts use paths relative to their own location — they all use hardcoded
absolute Windows paths (`c:\naren yashwanth N\...`) — so moving them into this folder
does not change their behavior; they'd still read/write the same files as before if run.

| File | What it did |
|---|---|
| `split_tabs.py` | Splits a "Tab N" - delimited text dump into individual files (writes into `class AH/`). |
| `split_open_source_essays.py` | Splits `OPEN_SOURCE_ESSAYS.txt` (essays from [openessays.org](https://www.openessays.org/)) into `class H/Tab_29.txt` onward — see `dataset/DATASET.md`'s "How each class was produced" for why this matters for provenance. |
| `create_hybrid_essays.py` | Early version of hybrid-essay generation via local Ollama (whole-essay rewrite approach) — superseded by `scripts/regenerate_hybrids.py`. |
| `create_hybrid_open_source_essays.py` | Same idea, applied to the openessays.org-sourced batch. |
| `move_hybrid_essays.py` | One-off file-move helper used while assembling `class AH/`. |
| `setup_ollama_model.py` | Interactive helper to `ollama pull` a model. |
| `test_ollama.py` | Connectivity smoke test for the local Ollama installation. |
| `HYBRID_ESSAY_README.txt` | Original usage notes for `create_hybrid_essays.py`, written before the openessays.org essays were added (still references only `Tab_1`-`Tab_29`). |
| `OPEN_SOURCE_ESSAYS.txt` | Raw text dump of essays from openessays.org — the source `split_open_source_essays.py` reads from. |
| `NLP.txt` | An earlier raw essay-text dump, same "Tab N" format as `OPEN_SOURCE_ESSAYS.txt`. No script in this repo currently references it; kept for audit rather than deleted, since its exact role in producing `class H/Tab_1`-`Tab_28` was not confirmed before archiving. |
| `project2` | Empty (0 bytes). Left as found rather than deleted. |
