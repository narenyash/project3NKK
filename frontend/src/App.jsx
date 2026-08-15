// App.jsx - the top-level component and the only piece of state management in this
// app (no Redux/Context - one screen, one component tree, plain useState is enough).
// Owns the full request lifecycle (idle -> loading -> results/error) and which sentence
// is currently hovered/pinned in the results view. Every other component in
// components/ is presentational - they receive data via props and call back up
// through onSubmit/onActivate/etc. rather than managing their own server state.
//
// Pulls in: api.js (the only place that talks to the backend), HowItWorks (the
// explainer shown above everything else), and one component per results-view section
// (EssayInput, LoadingState, FairnessBanner, EssaySummary, HighlightedEssay,
// SentenceDetailPanel, ErrorMessage).
import { useEffect, useState } from "react";
import { analyzeEssay, checkHealth, estimateWaitSeconds, ApiError } from "./api";
import HowItWorks from "./components/HowItWorks";
import EssayInput from "./components/EssayInput";
import LoadingState from "./components/LoadingState";
import FairnessBanner from "./components/FairnessBanner";
import EssaySummary from "./components/EssaySummary";
import HighlightedEssay from "./components/HighlightedEssay";
import SentenceDetailPanel from "./components/SentenceDetailPanel";
import ErrorMessage from "./components/ErrorMessage";
import "./App.css";

export default function App() {
  // One status enum drives which section of the page renders - see the JSX below.
  const [status, setStatus] = useState("idle"); // idle | loading | results | error
  const [result, setResult] = useState(null); // last successful /analyze response, as-is from the backend
  const [error, setError] = useState(null);
  const [estimatedSeconds, setEstimatedSeconds] = useState(0);
  const [backendUp, setBackendUp] = useState(null); // null = unknown yet

  // Which sentence's detail panel is showing. Hover shows it transiently; pinning
  // (click) keeps it open even after the mouse moves away - pinned always wins over
  // hovered when both are set, via the ?? fallback below.
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [pinnedIndex, setPinnedIndex] = useState(null);
  const activeIndex = pinnedIndex ?? hoveredIndex;

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    // Poll instead of checking once: if the backend is mid-restart (or comes up a
    // moment after this page loaded), a one-shot check would leave the app stuck
    // showing "unreachable" and the input permanently disabled until a manual reload.
    // Retry every 3s while down; stop polling once it's confirmed up.
    async function poll() {
      const up = await checkHealth();
      if (cancelled) return;
      setBackendUp(up);
      if (!up) {
        timer = setTimeout(poll, 3000);
      }
    }
    poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  // Fires the actual /analyze request (via api.js). Resets any pinned/hovered sentence
  // from a previous result so the detail panel doesn't show stale data mid-request.
  async function handleSubmit(text, wordCount) {
    setStatus("loading");
    setError(null);
    setEstimatedSeconds(estimateWaitSeconds(wordCount));
    setPinnedIndex(null);
    setHoveredIndex(null);
    try {
      const data = await analyzeEssay(text);
      setResult(data);
      setStatus("results");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong analyzing this essay.");
      setStatus("error");
    }
  }

  // "Analyze another essay" - back to the empty input state, discarding the last result.
  function handleReset() {
    setStatus("idle");
    setResult(null);
    setError(null);
    setPinnedIndex(null);
    setHoveredIndex(null);
  }

  // Passed down to HighlightedEssay as onActivate, called on both hover and click.
  // Clicking an already-pinned sentence unpins it (toggle), matching the (i) icon and
  // sentence-span click behavior described in HighlightedEssay.jsx.
  function handleActivate(index, kind) {
    if (kind === "hover") setHoveredIndex(index);
    if (kind === "pin") setPinnedIndex((prev) => (prev === index ? null : index));
  }

  // Only "hover" is ever deactivated this way - a pin is cleared explicitly (via
  // handleActivate's toggle, or SentenceDetailPanel's close button), never by mouseleave.
  function handleDeactivate(kind) {
    if (kind === "hover") setHoveredIndex(null);
  }

  const activeSentence = result && activeIndex !== null ? result.sentences[activeIndex] : null;

  return (
    <div className="app">
      <HowItWorks />

      <header className="app-header">
        <h1>AI Essay Detector</h1>
        <p className="app-tagline">
          Statistical analysis of essay text — not a verdict. See the fairness note below
          before drawing conclusions.
        </p>
      </header>

      {backendUp === false && (
        <ErrorMessage message="Can't reach the analysis service right now. Make sure the backend is running, then reload this page." />
      )}

      {status === "idle" && <EssayInput onSubmit={handleSubmit} disabled={backendUp === false} />}

      {status === "loading" && (
        <>
          <EssayInput onSubmit={() => {}} disabled />
          <LoadingState estimatedSeconds={estimatedSeconds} />
        </>
      )}

      {status === "error" && <ErrorMessage message={error} onDismiss={handleReset} />}

      {status === "results" && result && (
        <div className="results-layout">
          <div className="results-main">
            <FairnessBanner note={result.fairness_note} />
            <EssaySummary essayLevel={result.essay_level} />
            <div className="card essay-card">
              <HighlightedEssay
                sentences={result.sentences}
                paragraphs={result.paragraphs}
                paragraphOutlierNote={result.paragraph_outlier_note}
                activeIndex={activeIndex}
                onActivate={handleActivate}
                onDeactivate={handleDeactivate}
              />
            </div>
            <button className="btn-secondary" onClick={handleReset}>
              Analyze another essay
            </button>
          </div>
          <div className="results-side">
            <SentenceDetailPanel
              sentence={activeSentence}
              pinned={pinnedIndex !== null}
              onClose={() => setPinnedIndex(null)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
