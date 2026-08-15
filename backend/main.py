"""
Step 4: the FastAPI service. POST /analyze runs the full inference pipeline; GET /health
confirms everything is loaded. Models/GPT-2/spaCy/baseline are loaded once at import time
(via model_loader.get_state()), not per-request - see model_loader.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import get_state
from pipeline import analyze_essay, MIN_WORDS, MAX_WORDS
from text_utils import word_count

app = FastAPI(title="AI Essay Detector API", version="1.0.0")

# Local dev only: the frontend runs on a different origin/port (Vite dev server), so the
# browser blocks fetch() to this API without CORS headers. Caught via a real
# browser-driven integration test (Playwright), not the earlier Python `requests`-based
# test, which bypasses CORS entirely since it isn't a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load everything once, at import time (process startup), not per-request.
_STATE = get_state()


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Essay text to analyze.")


# Polled by frontend/src/api.js's checkHealth() (App.jsx retries every 3s while this
# reports not-ready) - lets the frontend distinguish "backend unreachable" from
# "backend up but still loading GPT-2/models," which otherwise look identical from the
# browser's side.
@app.get("/health")
def health():
    ready = _STATE.ready()
    return {
        "status": "ok" if ready else "not_ready",
        "models_loaded": ready,
    }


# The only real endpoint: validates the request (non-empty, within the word-count
# bounds pipeline.py's models were trained/evaluated on), then delegates everything
# else to analyze_essay(). Every 400 response here has a specific, human-readable
# reason rather than a generic validation error.
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    wc = word_count(text)
    if wc < MIN_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too short ({wc} words). Minimum {MIN_WORDS} words required - "
                   f"both models were trained on essay-length text (typically several "
                   f"hundred words), and the essay-level features (burstiness, repetition) "
                   f"need multiple sentences to be meaningful at all.",
        )
    if wc > MAX_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too long ({wc} words). Maximum {MAX_WORDS} words - longer "
                   f"inputs run one GPT-2 forward pass per sentence and can take a long "
                   f"time; this cap keeps response times reasonable.",
        )

    try:
        result = analyze_essay(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result
