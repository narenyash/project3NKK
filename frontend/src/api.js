// api.js - the frontend's only point of contact with the FastAPI backend
// (backend/main.py). Every network call the app makes goes through one of the three
// functions below: checkHealth() (used by App.jsx on load, polled until the backend is
// up), analyzeEssay() (the actual essay-analysis request), and estimateWaitSeconds()
// (a client-side estimate shown in the loading state, not a real network call).
// Pulls in nothing beyond the browser's built-in fetch/AbortSignal - no HTTP library.

const BASE_URL = "http://127.0.0.1:8000";

// Thrown by analyzeEssay() on any failure (network error or a non-2xx response) so
// callers (App.jsx) can show a message without needing to know fetch's own error shapes.
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// Pings GET /health. Returns true only when the backend is reachable AND has finished
// loading GPT-2 + both trained models (models_loaded: true) - a backend that's up but
// still mid-startup should still read as "not ready yet," not "ready."
export async function checkHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === "ok" && data.models_loaded === true;
  } catch {
    return false;
  }
}

// POSTs essay text to /analyze and returns the parsed JSON result (see
// backend/README.md for the full response shape: essay_level, sentences, paragraphs,
// fairness_note, etc.). Throws ApiError on any failure - a network-level failure (backend
// unreachable) and a backend-returned error (e.g. "too short") both surface the same way
// to the caller, just with different messages/status codes.
export async function analyzeEssay(text) {
  let res;
  try {
    res = await fetch(`${BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (err) {
    throw new ApiError(
      "Could not reach the analysis service. Make sure the backend is running at " +
        `${BASE_URL} and try again.`,
      0
    );
  }

  if (!res.ok) {
    let detail = "The analysis service returned an error.";
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      // ignore parse failure, use default message
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}

// Response-time estimate, based on the backend integration test in backend/README.md:
// roughly 0.15-0.22s per sentence, with ~20 words/sentence average across the essays
// tested (h_006: 7.17s/33 sentences, ai_001: 33.97s/228 sentences, hy_012: 12.90s/57
// sentences - about 0.15-0.24s per sentence across all three). Used only to set
// expectations in the loading UI, not as a hard promise.
const SECONDS_PER_SENTENCE_ESTIMATE = 0.2;
const WORDS_PER_SENTENCE_ESTIMATE = 20;

export function estimateWaitSeconds(wordCount) {
  const estimatedSentences = Math.max(1, Math.round(wordCount / WORDS_PER_SENTENCE_ESTIMATE));
  return Math.round(estimatedSentences * SECONDS_PER_SENTENCE_ESTIMATE);
}
