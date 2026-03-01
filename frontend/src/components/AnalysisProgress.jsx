import { Loader2 } from "lucide-react";
import "./AnalysisProgress.css";

export function AnalysisProgress({ status }) {
  // Translate backend status to user-friendly messages
  const getStatusMessage = () => {
    switch (status) {
      case "QUEUED":
        return "Waiting in queue...";
      case "PROCESSING":
        return "Analyzing video frames...";
      default:
        return "Connecting to backend...";
    }
  };

  return (
    <div className="panel progress-panel">
      <div className="progress-spinner-container">
        <Loader2 size={48} className="progress-spinner" />
      </div>
      <h3 className="progress-title">AI Detection in Progress</h3>
      <p className="progress-status">{getStatusMessage()}</p>

      <div className="progress-bar-container">
        <div className="progress-bar-indeterminate"></div>
      </div>

      <p className="progress-hint">
        This process evaluates spatial and temporal artifacts using our
        dual-model architecture. It may take up to a minute depending on video
        length.
      </p>
    </div>
  );
}
