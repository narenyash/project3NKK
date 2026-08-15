// EssaySummary.jsx - the top-of-results card: the big headline percentage
// (essay_level.ai_likelihood_score from the API response), a plain-language
// description of it, the backend's confidence note, and the legend explaining every
// visual signal used elsewhere on the page (highlight, underline, paragraph average,
// outlier badge). Purely presentational - all of its input is the essayLevel object
// straight from the /analyze response, no local state.
import { ESSAY_HIGHLIGHT_CUTOFF } from "../constants/thresholds";

// Deliberately descriptive, never a verdict. "Shows patterns statistically associated
// with..." not "This essay IS...". Hard constraint #1, carried through from the backend.
function describeLikelihood(score) {
  if (score === null || score === undefined) return "Unable to compute a score for this essay.";
  if (score < 0.3) return "Shows few patterns statistically associated with AI-written text.";
  if (score < 0.7) return "Shows a mix of patterns, some associated with AI-written text.";
  return "Shows patterns strongly and consistently associated with AI-written text.";
}

export default function EssaySummary({ essayLevel }) {
  const score = essayLevel.ai_likelihood_score;
  const pct = score !== null && score !== undefined ? Math.round(score * 100) : null;

  return (
    <div className="card summary-card">
      <div className="summary-score-row">
        <div className="summary-score-value" aria-label="AI-likelihood score">
          {pct !== null ? `${pct}%` : "—"}
        </div>
        <div className="summary-score-text">
          <p className="summary-score-description">{describeLikelihood(score)}</p>
          <p className="summary-confidence-note">{essayLevel.confidence_note}</p>
        </div>
      </div>

      <div className="legend">
        <div className="legend-item">
          <span className="legend-swatch legend-swatch-red" aria-hidden="true" />
          <span>
            <strong>Highlighted sentences</strong> — this is the main signal: sentences
            scoring at or above {Math.round(ESSAY_HIGHLIGHT_CUTOFF * 100)}% on the
            essay-wide AI-likelihood measure above. The exact percentage is shown next to
            each one. Click any sentence, or its ⓘ, for the full reasoning behind it.
          </span>
        </div>
        <div className="legend-item">
          <span className="legend-swatch legend-swatch-orange" aria-hidden="true" />
          <span>
            <strong>Underline</strong> — a separate, sentence-only check (darker/thicker =
            stronger). Many sentences won't lean clearly either way when looked at on their
            own — that's expected, not a malfunction.
          </span>
        </div>
        <div className="legend-item">
          <span className="legend-swatch legend-swatch-red" aria-hidden="true" />
          <span>
            <strong>Paragraph average</strong> — shown above each paragraph, this is the
            mean of that paragraph's sentence scores. A paragraph can read as AI-associated
            in aggregate even when no single sentence in it is individually extreme.
          </span>
        </div>
        <div className="legend-item">
          <span className="legend-swatch legend-swatch-orange" aria-hidden="true" />
          <span>
            <strong>⚠ Statistically unusual paragraph</strong> — an exploratory,
            unvalidated heuristic (not run through either trained model) comparing a
            paragraph's own writing statistics only against the other paragraphs in this
            same essay. A flag means "different from the rest of this essay," not "AI" —
            click the badge for why.
          </span>
        </div>
      </div>
    </div>
  );
}
