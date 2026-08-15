// ErrorMessage.jsx - generic error/alert card, used for two distinct cases in App.jsx:
// the backend being unreachable (no onDismiss - there's nothing to dismiss back to) and
// a failed /analyze request (onDismiss resets back to the input screen). Purely
// presentational, no state or API calls of its own.
export default function ErrorMessage({ message, onDismiss }) {
  return (
    <div className="card error-card" role="alert">
      <p>{message}</p>
      {onDismiss && (
        <button className="btn-secondary" onClick={onDismiss}>
          Try again
        </button>
      )}
    </div>
  );
}
