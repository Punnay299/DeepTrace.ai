import { AlertCircle } from "lucide-react";
import "./ErrorBanner.css";

export function ErrorBanner({ message, onDismiss }) {
  if (!message) return null;

  return (
    <div className="error-banner">
      <div className="error-content">
        <AlertCircle size={20} className="error-icon" />
        <span className="error-message">{message}</span>
      </div>
      {onDismiss && (
        <button onClick={onDismiss} className="error-dismiss">
          ×
        </button>
      )}
    </div>
  );
}
