// Plain-English translations for the 7 raw feature names the essay-level model uses
// (the sentence-level model reuses the first 2 of these). Every feature carries a short
// NAME (used as a heading, so a reader can see exactly which of the 7 measures they're
// looking at) plus a longer DESCRIPTION of what it actually measures, and an explicit
// "Essay-wide:" or "This sentence:" prefix.
//
// The prefix is determined by what the FEATURE actually measures, not which model
// surfaced it: perplexity_z and logprob_variance_z are genuinely per-sentence
// properties (Phase 2 finding) even when they appear in top_features (the essay
// model's list, which also includes 5 essay-wide constants repeated on every
// sentence) - so they always get "This sentence:", regardless of source.

const SENTENCE_LEVEL_FEATURE_NAMES = new Set(["perplexity_z", "logprob_variance_z"]);

// The 7 measures used by the essay-level model, in a fixed reference order (not sorted
// by contribution) - used to give a reader the full named list up front.
export const ALL_ESSAY_FEATURES = [
  "perplexity_z",
  "logprob_variance_z",
  "sentence_length_variance_z",
  "syntactic_depth_variance_z",
  "perplexity_variance_z",
  "ngram_repeat_rate_z",
  "transition_repetition_rate_z",
];

const FEATURE_NAME = {
  perplexity_z: "Word predictability",
  logprob_variance_z: "Predictability swings",
  sentence_length_variance_z: "Sentence-length variation",
  syntactic_depth_variance_z: "Grammatical-complexity variation",
  perplexity_variance_z: "Word-choice variation",
  ngram_repeat_rate_z: "Phrase repetition",
  transition_repetition_rate_z: "Transition-word repetition",
};

const FEATURE_TEXT = {
  perplexity_z:
    "How unusual or unpredictable this sentence's wording is, compared to typical human essay writing",
  logprob_variance_z:
    "How much this sentence's word-by-word predictability swings between predictable and surprising",
  sentence_length_variance_z: "How much sentence length varies across the whole essay",
  syntactic_depth_variance_z: "How much sentence grammatical complexity varies across the whole essay",
  perplexity_variance_z: "How much word-choice predictability varies from sentence to sentence, essay-wide",
  ngram_repeat_rate_z: "How much repeated phrasing appears across the whole essay",
  transition_repetition_rate_z:
    'How often the same transition word (e.g. "however," "moreover") repeats across the whole essay',
};

export function featureLabel(featureName) {
  const isSentenceLevel = SENTENCE_LEVEL_FEATURE_NAMES.has(featureName);
  return {
    prefix: isSentenceLevel ? "This sentence:" : "Essay-wide:",
    name: FEATURE_NAME[featureName] || featureName,
    text: FEATURE_TEXT[featureName] || featureName,
  };
}

// Linear clamp of the z-score into 0-10: 0 = exactly typical for human writing (z = 0),
// 10 = as extreme as the scale tracks (|z| >= 3). Same honest, non-distorted linear
// mapping used for the highlighting, just relabeled into units anyone can read.
export function scoreOutOf10(zValue) {
  const absZ = Math.min(Math.abs(zValue), 3);
  return Math.round((absZ / 3) * 100) / 10;
}

function magnitudePhrase(score10) {
  if (score10 > 5) return "far more than typical for human writing";
  if (score10 > 1.7) return "somewhat more than typical for human writing";
  return "about typical for human writing";
}

const DIRECTION_LABEL = {
  toward_ai: "Leans AI",
  toward_human: "Leans human",
  neutral: "Neutral",
};

const DIRECTION_PHRASE = {
  toward_ai: "a pattern more often seen in AI-written text",
  toward_human: "a pattern more often seen in human writing",
  neutral: "about equally common in both, so this factor doesn't move the needle much here",
};

// Raw values come straight from feature extraction (before z-scoring) and are mostly
// unitless model-internal numbers; the two "rate" features are genuinely 0-1 ratios, so
// those read far more clearly as a percentage than as a raw decimal.
const PERCENT_FEATURES = new Set(["ngram_repeat_rate_z", "transition_repetition_rate_z"]);

function formatRawValue(featureName, rawValue) {
  if (rawValue === null || rawValue === undefined) return null;
  if (PERCENT_FEATURES.has(featureName)) return `${(rawValue * 100).toFixed(1)}%`;
  return rawValue.toFixed(2);
}

// Structured version (not a single prose sentence) so the detail panel can render each
// factor as a small, scannable card: a named heading, a plain description, a 0-10
// score with its direction, and the actual measured value.
export function buildFactorDetail(featureName, zValue, direction, rawValue) {
  const { prefix, name, text } = featureLabel(featureName);
  const score10 = scoreOutOf10(zValue);
  return {
    prefix,
    name,
    description: text,
    score10,
    magnitude: magnitudePhrase(score10),
    directionKey: direction in DIRECTION_LABEL ? direction : "neutral",
    directionLabel: DIRECTION_LABEL[direction] || DIRECTION_LABEL.neutral,
    directionPhrase: DIRECTION_PHRASE[direction] || DIRECTION_PHRASE.neutral,
    rawText: formatRawValue(featureName, rawValue),
  };
}
