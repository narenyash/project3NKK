// EssayInput.jsx - the textarea + submit form shown in the "idle" and "loading" states.
// Owns only its own draft text and the derived word count/validity - the actual submit
// (calling the backend) happens in App.jsx's handleSubmit, reached via onSubmit(text,
// count). MIN_WORDS/MAX_WORDS here are the frontend's own pre-check, purely so a user
// gets instant feedback while typing - the backend (backend/pipeline.py's MIN_WORDS/
// MAX_WORDS) enforces the same bounds again server-side regardless, so these two
// constants must be kept in sync manually if either changes.
import { useState } from "react";

const MIN_WORDS = 50;
const MAX_WORDS = 5000;

// Word count via whitespace-splitting, same simple definition used throughout the
// project (see scripts/text_utils.py's word_count() for the backend's identical logic).
function wordCount(text) {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

export default function EssayInput({ onSubmit, disabled }) {
  const [text, setText] = useState("");
  const count = wordCount(text);
  const tooShort = count > 0 && count < MIN_WORDS;
  const tooLong = count > MAX_WORDS;
  const canSubmit = count >= MIN_WORDS && count <= MAX_WORDS && !disabled;

  // Guards against submitting out-of-range text even if the disabled button is somehow
  // bypassed (e.g. pressing Enter in the textarea) - canSubmit is the single source of
  // truth for validity, checked here again rather than trusted from the button state.
  function handleSubmit(e) {
    e.preventDefault();
    if (canSubmit) onSubmit(text, count);
  }

  return (
    <form className="card input-card" onSubmit={handleSubmit}>
      <h2 className="card-title">Analyze an essay</h2>
      <p className="card-subtitle">
        Paste essay text below. Works best on full essays (a few hundred words or more) —
        the underlying models were trained and evaluated on essay-length text, not single
        sentences.
      </p>
      <textarea
        className="essay-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste an essay here..."
        rows={14}
        disabled={disabled}
      />
      <div className="input-footer">
        <span className={`word-count ${tooShort || tooLong ? "word-count-warn" : ""}`}>
          {count.toLocaleString()} word{count === 1 ? "" : "s"}
          {tooShort && ` — needs at least ${MIN_WORDS}`}
          {tooLong && ` — over the ${MAX_WORDS.toLocaleString()} word limit`}
        </span>
        <button type="submit" className="btn-primary" disabled={!canSubmit}>
          {disabled ? "Analyzing…" : "Analyze essay"}
        </button>
      </div>
    </form>
  );
}
