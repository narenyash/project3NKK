// LoadingState.jsx - shown while App.jsx awaits the /analyze response. Runs its own
// elapsed-time ticker (independent of the actual request) purely for user feedback -
// estimatedSeconds comes from api.js's estimateWaitSeconds(), a rough client-side guess,
// not a value the backend returns.
import { useEffect, useState } from "react";

export default function LoadingState({ estimatedSeconds }) {
  const [elapsed, setElapsed] = useState(0);

  // Ticks once per 500ms from mount until unmount (the whole loading state's lifetime,
  // since this component only exists while status === "loading" in App.jsx).
  useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 500);
    return () => clearInterval(interval);
  }, []);

  const overEstimate = estimatedSeconds > 0 && elapsed > estimatedSeconds * 1.3;

  return (
    <div className="card loading-card" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p className="loading-title">Analyzing essay…</p>
      <p className="loading-detail">
        {estimatedSeconds > 0
          ? `Estimated wait: ~${estimatedSeconds}s. `
          : ""}
        Longer essays take longer — each sentence runs its own language-model pass, so a
        long essay can take up to 30+ seconds.
        {overEstimate && " This one's taking a bit longer than estimated, still working…"}
      </p>
      <p className="loading-elapsed">{elapsed}s elapsed</p>
    </div>
  );
}
