// HighlightedEssay.jsx - the main results view: the essay text itself, grouped into
// paragraphs, with three independent visual signals layered on (see the two comment
// blocks below for what each one means and where its data comes from). This is the
// single most visually complex component in the app - App.jsx only passes it data and
// two callbacks (onActivate/onDeactivate); all hover/pin/pop/expand interaction state
// lives locally here via useState.
//
// Pulls in: constants/thresholds.js for the shared color/cutoff logic (kept there, not
// here, so SentenceDetailPanel's "worth noticing" logic and this component's highlight
// logic can never drift apart).
import { useState } from "react";
import { essayGradientStyle, sentenceUnderlineStyle, isNoticeable, ESSAY_HIGHLIGHT_CUTOFF } from "../constants/thresholds";

const POP_DURATION_MS = 320;

// One sentence's rendering: the highlighted/underlined span itself, its inline %
// badge (only shown once the score clears isNoticeable's bar), and its ⓘ button.
// Split out from the main component purely to keep the paragraph-grouping .map() below
// readable - it has no state of its own, everything comes from props.
function SentenceSpan({ s, i, isLast, isActive, isPopping, onActivate, onDeactivate, onInfoClick }) {
  const essayScore = s.essay_model_local_score;
  const style = {
    ...essayGradientStyle(essayScore),
    ...sentenceUnderlineStyle(s.sentence_level_score),
  };
  const classes = ["sentence-span"];
  if (isActive) classes.push("sentence-active");
  const pct = essayScore !== null && essayScore !== undefined ? Math.round(essayScore * 100) : null;

  return (
    <span className="sentence-wrap">
      <span
        className={classes.join(" ")}
        style={style}
        tabIndex={0}
        role="button"
        aria-label={`Sentence ${i + 1}${pct !== null ? `, ${pct}% AI-likelihood contribution` : ""}`}
        onMouseEnter={() => onActivate(i, "hover")}
        onMouseLeave={() => onDeactivate("hover")}
        onFocus={() => onActivate(i, "hover")}
        onBlur={() => onDeactivate("hover")}
        onClick={() => onActivate(i, "pin")}
      >
        {s.text}
      </span>
      {isNoticeable(essayScore) && <span className="inline-pct">{pct}%</span>}
      <button
        type="button"
        className={`info-icon${isPopping ? " info-icon-pop" : ""}`}
        aria-label={`Why this sentence is highlighted: sentence ${i + 1}`}
        onClick={() => onInfoClick(i)}
      >
        ⓘ
      </button>
      {!isLast ? " " : ""}
    </span>
  );
}

// Renders the essay grouped into paragraphs (blank-line-separated blocks in the original
// submission, per backend/pipeline.py's paragraph_index) - not a list of separate sentence
// boxes within each paragraph. Each paragraph gets its own small header showing the
// paragraph-level rollup score, so a reader can see "this whole paragraph reads as
// AI-associated" even when no single sentence in it is individually extreme.
//
// Background color (primary, essay-level 7-feature score) is a solid highlight above
// ESSAY_HIGHLIGHT_CUTOFF, nothing below it - a clear, high-contrast look rather than a
// tint that fades into invisibility. Underline (secondary, sentence-level 2-feature
// score) stays continuously proportional, no cutoff, and visibly lower-weight than the
// primary highlight.
export default function HighlightedEssay({
  sentences,
  paragraphs,
  paragraphOutlierNote,
  activeIndex,
  onActivate,
  onDeactivate,
}) {
  const [poppingIndex, setPoppingIndex] = useState(null); // which sentence's ⓘ is mid-animation
  const [expandedOutlier, setExpandedOutlier] = useState(null); // which paragraph's outlier note is open (one at a time)

  // Clicking ⓘ both opens the detail panel (via onActivate, same as clicking the
  // sentence itself) and triggers the pop animation - poppingIndex is cleared after
  // POP_DURATION_MS via the CSS animation's real duration, not tied to the panel state.
  function handleInfoClick(i) {
    setPoppingIndex(i);
    onActivate(i, "pin");
    setTimeout(() => setPoppingIndex((prev) => (prev === i ? null : prev)), POP_DURATION_MS);
  }

  // Toggles one paragraph's outlier note open/closed; opening a new one implicitly
  // closes whichever was previously open, since expandedOutlier only holds one index.
  function toggleOutlier(pIndex) {
    setExpandedOutlier((prev) => (prev === pIndex ? null : pIndex));
  }

  // Regroup the flat sentences array (as received from the API) by paragraph_index.
  // Relies on paragraph_index being non-decreasing in the original array (guaranteed by
  // backend/pipeline.py's _assign_paragraphs) so each paragraph's sentences stay in
  // their original reading order within the Map.
  const groups = new Map();
  sentences.forEach((s, i) => {
    const p = s.paragraph_index ?? 0;
    if (!groups.has(p)) groups.set(p, []);
    groups.get(p).push({ s, i });
  });

  return (
    <div className="essay-text">
      {[...groups.entries()].map(([pIndex, members]) => {
        const meta = paragraphs && paragraphs[pIndex];
        const pScore = meta ? meta.ai_likelihood_score : null;
        const pPct = pScore !== null && pScore !== undefined ? Math.round(pScore * 100) : null;
        const notable = pScore !== null && pScore !== undefined && pScore >= ESSAY_HIGHLIGHT_CUTOFF;
        const outlier = meta && meta.local_outlier;

        return (
          <div key={pIndex} className={`paragraph-block${notable ? " paragraph-block-notable" : ""}`}>
            <div className="paragraph-header">
              <span className="paragraph-label">Paragraph {pIndex + 1}</span>
              {pPct !== null && (
                <span className="paragraph-score">{pPct}% AI-likelihood (paragraph average)</span>
              )}
              {outlier && outlier.notable && (
                <button
                  type="button"
                  className="outlier-badge"
                  aria-label={`Paragraph ${pIndex + 1} is a statistical outlier within this essay - unvalidated heuristic, click for details`}
                  onClick={() => toggleOutlier(pIndex)}
                >
                  ⚠ statistically unusual paragraph
                </button>
              )}
            </div>
            {outlier && outlier.notable && expandedOutlier === pIndex && (
              <div className="outlier-note">
                <ul className="outlier-deviation-list">
                  {outlier.mean_perplexity_deviation !== null && (
                    <li>Mean perplexity deviation from this essay's other paragraphs: {outlier.mean_perplexity_deviation.toFixed(2)}</li>
                  )}
                  {outlier.sentence_length_variance_deviation !== null && (
                    <li>Sentence-length variance deviation: {outlier.sentence_length_variance_deviation.toFixed(2)}</li>
                  )}
                  {outlier.syntactic_depth_variance_deviation !== null && (
                    <li>Grammatical-complexity variance deviation: {outlier.syntactic_depth_variance_deviation.toFixed(2)}</li>
                  )}
                </ul>
                <p className="outlier-caveat">{paragraphOutlierNote}</p>
              </div>
            )}
            <p className="paragraph-sentences">
              {members.map(({ s, i }, memberIdx) => (
                <SentenceSpan
                  key={s.index}
                  s={s}
                  i={i}
                  isLast={memberIdx === members.length - 1}
                  isActive={activeIndex === i}
                  isPopping={poppingIndex === i}
                  onActivate={onActivate}
                  onDeactivate={onDeactivate}
                  onInfoClick={handleInfoClick}
                />
              ))}
            </p>
          </div>
        );
      })}
    </div>
  );
}
