// Persistent, visible banner - never a modal, never collapsed by default. Direct
// carry-over of the backend's fairness_note field, per the explicit hard constraint
// that this must be seen without extra clicks.
export default function FairnessBanner({ note }) {
  if (!note) return null;
  return (
    <div className="fairness-banner" role="note">
      <span className="fairness-icon" aria-hidden="true">
        ⚠
      </span>
      <p>{note}</p>
    </div>
  );
}
