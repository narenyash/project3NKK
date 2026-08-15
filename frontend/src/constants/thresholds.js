// Highlighting rules for the essay view.
//
// essay_model_local_score (the essay-level model's 7-feature per-sentence output, before
// averaging into the essay-level verdict) is the PRIMARY signal: sentences scoring above
// ESSAY_HIGHLIGHT_CUTOFF get one clearly visible, solid highlight color; sentences below
// it get none. This is a deliberately high-contrast look (confirmed with the user against
// a reference example) rather than a continuously-fading tint, which read as "not
// highlighted" for anything short of very high scores.
//
// sentence_level_score (the 2-feature sentence-only model) is SECONDARY: a thin
// underline, still linearly proportional to the score (no cutoff), kept visibly lower
// weight than the primary highlight so it reads as a supporting annotation, not a
// competing verdict.
//
// The cutoff is deliberately much lower than the old RED_THRESHOLD (0.85) - that
// threshold gated out any essay short of extreme confidence, which is what made
// moderate-confidence essays show zero highlighted sentences. essay_model_local_score
// has real spread across sentences (15-96% observed in testing), so a ~40% bar surfaces
// the sentences that actually pulled the essay's score up without requiring near-certainty.
export const ESSAY_HIGHLIGHT_CUTOFF = 0.4;

const ESSAY_RED_RGB = "178, 58, 46";
const SENTENCE_AMBER_RGB = "179, 122, 20";

const ESSAY_HIGHLIGHT_OPACITY = 0.28; // one fixed, clearly visible shade - not proportional
const SENTENCE_MAX_UNDERLINE_OPACITY = 0.55;
const SENTENCE_MAX_UNDERLINE_WIDTH = 3; // px, at score=1

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

export function essayGradientStyle(essayLocalScore) {
  if (essayLocalScore === null || essayLocalScore === undefined) return {};
  if (essayLocalScore < ESSAY_HIGHLIGHT_CUTOFF) return {};
  return { backgroundColor: `rgba(${ESSAY_RED_RGB}, ${ESSAY_HIGHLIGHT_OPACITY})` };
}

export function sentenceUnderlineStyle(sentenceScore) {
  if (sentenceScore === null || sentenceScore === undefined) return {};
  const t = clamp01(sentenceScore);
  const opacity = t * SENTENCE_MAX_UNDERLINE_OPACITY;
  const width = Math.max(1, t * SENTENCE_MAX_UNDERLINE_WIDTH);
  return {
    borderBottom: `${width.toFixed(1)}px solid rgba(${SENTENCE_AMBER_RGB}, ${opacity.toFixed(3)})`,
  };
}

// Show the inline essay-level percentage badge on the same sentences that get the solid
// highlight - keeps the exact number visible without requiring the panel to be open.
export function isNoticeable(score) {
  return score !== null && score !== undefined && score >= ESSAY_HIGHLIGHT_CUTOFF;
}
