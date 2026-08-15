// SentenceDetailPanel.jsx - the sidebar shown for whichever sentence is currently
// active (hovered or pinned - see App.jsx's activeIndex). Two-tier layout mirrors the
// backend's own primary/secondary split: the essay-level (7-feature) contribution on
// top in full detail, the sentence-level (2-feature) check below, visually smaller and
// explicitly labeled "secondary." Every number shown here comes straight from one
// sentence object in the /analyze response - this component does no computation of its
// own beyond formatting, deferring all "what does this number mean" logic to
// constants/featureLabels.js's buildFactorDetail().
import { buildFactorDetail, featureLabel, ALL_ESSAY_FEATURES } from "../constants/featureLabels";

// How close sentence_level_score has to be to 0.5 to trigger the "this is genuine
// uncertainty, not a broken display" note below - see constants/featureLabels.js's
// module comment for why that model clusters near 50% so often.
const NEAR_HALF_BAND = 0.05;

// One factor's card (either an essay-model or sentence-model feature) - name,
// direction badge, plain-language description, 0-10 score, and the raw measured value.
// Shared by both sections below since essay-model and sentence-model features use the
// identical {feature, value, direction, raw_value} shape from the backend.
function FeatureCard({ feature }) {
  const d = buildFactorDetail(feature.feature, feature.value, feature.direction, feature.raw_value);
  return (
    <li className={`feature-card feature-card-${d.directionKey}`}>
      <div className="feature-card-header">
        <span className="feature-card-prefix">{d.prefix}</span>
        <span className="feature-card-name">{d.name}</span>
        <span className={`feature-card-badge feature-card-badge-${d.directionKey}`}>{d.directionLabel}</span>
      </div>
      <p className="feature-card-desc">{d.description}</p>
      <p className="feature-card-meta">
        <span className="feature-card-score">{d.score10}/10 unusual</span> ({d.magnitude}) —{" "}
        {d.directionPhrase}.{" "}
        {d.rawText !== null ? (
          <span className="feature-card-raw">Actual value: {d.rawText}</span>
        ) : (
          <span className="feature-card-raw feature-card-raw-missing">
            Actual value unavailable for this sentence.
          </span>
        )}
      </p>
    </li>
  );
}

export default function SentenceDetailPanel({ sentence, onClose, pinned }) {
  if (!sentence) {
    return (
      <aside className="card detail-card detail-empty">
        <p>Click a highlighted sentence or its ⓘ icon to see the numbers behind it.</p>
      </aside>
    );
  }

  const essayLocal = sentence.essay_model_local_score;
  const sentScore = sentence.sentence_level_score;
  const nearHalf = sentScore !== null && sentScore !== undefined && Math.abs(sentScore - 0.5) <= NEAR_HALF_BAND;

  return (
    <aside className="card detail-card" aria-live="polite">
      {pinned && (
        <button className="detail-close" onClick={onClose} aria-label="Close detail panel">
          ×
        </button>
      )}
      <p className="detail-sentence-text">"{sentence.text}"</p>

      <div className="detail-section detail-section-primary">
        <div className="detail-section-header">
          <span>This sentence's AI-likelihood contribution</span>
          <span className="detail-score detail-score-primary">
            {essayLocal !== null && essayLocal !== undefined ? `${Math.round(essayLocal * 100)}%` : "—"}
          </span>
        </div>
        <p className="detail-note">{sentence.essay_model_local_score_note}</p>
        <p className="detail-metrics-summary">
          Measured using 7 factors:{" "}
          {ALL_ESSAY_FEATURES.map((f, i) => (
            <span key={f}>
              {featureLabel(f).name}
              {i < ALL_ESSAY_FEATURES.length - 1 ? ", " : "."}
            </span>
          ))}
        </p>
        <ul className="feature-list">
          {sentence.top_features.map((f) => (
            <FeatureCard key={f.feature} feature={f} />
          ))}
        </ul>
      </div>

      <div className="detail-section detail-section-secondary">
        <div className="detail-section-header">
          <span>Sentence-only check (secondary signal)</span>
          <span className="detail-score">
            {sentScore !== null && sentScore !== undefined ? `${Math.round(sentScore * 100)}%` : "—"}
          </span>
        </div>
        {nearHalf && (
          <p className="detail-note detail-note-near-half">
            Looked at on its own, this sentence doesn't lean clearly toward either AI or
            human writing — that's a normal, common result for an individual sentence and
            simply means nothing stood out here.
          </p>
        )}
        <p className="detail-note">{sentence.confidence_note}</p>
        <ul className="feature-list">
          {sentence.sentence_model_features.map((f) => (
            <FeatureCard key={f.feature} feature={f} />
          ))}
        </ul>
      </div>
    </aside>
  );
}
