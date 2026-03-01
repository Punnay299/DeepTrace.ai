import { useState, useRef, useEffect } from "react";
import { ShieldAlert } from "lucide-react";
import { apiClient } from "./api/client";
import { ErrorBanner } from "./components/ErrorBanner";
import { VideoUploader } from "./components/VideoUploader";
import { AnalysisProgress } from "./components/AnalysisProgress";
import { ResultsDashboard } from "./components/ResultsDashboard";
import "./App.css";

const PHASE = {
  IDLE: "IDLE",
  ANALYZING: "ANALYZING",
  COMPLETE: "COMPLETE",
  ERROR: "ERROR",
};

const MAX_POLLS = 60; // 60 retries * 2.5s = ~150 seconds (2.5 minutes)
const POLL_INTERVAL_MS = 2500;

function App() {
  const [phase, setPhase] = useState(PHASE.IDLE);
  const [globalError, setGlobalError] = useState(null);
  const [currentJob, setCurrentJob] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState("");
  const [resultData, setResultData] = useState(null);

  const pollCountRef = useRef(0);
  const pollTimerRef = useRef(null);

  useEffect(() => {
    return () => stopPolling();
  }, []);

  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const handleUpload = async (file) => {
    try {
      setGlobalError(null);
      setPhase(PHASE.ANALYZING);
      setAnalysisStatus("QUEUED");

      const response = await apiClient.uploadVideo(file);
      setCurrentJob(response.job_id);

      // Begin strictly capped polling sequence
      pollCountRef.current = 0;
      pollTimerRef.current = setInterval(
        () => checkJobStatus(response.job_id),
        POLL_INTERVAL_MS,
      );
    } catch (err) {
      handleError(err);
    }
  };

  const checkJobStatus = async (jobId) => {
    try {
      pollCountRef.current += 1;

      if (pollCountRef.current > MAX_POLLS) {
        stopPolling();
        throw new Error("Analysis timed out. Please try again.");
      }

      const response = await apiClient.checkStatus(jobId);

      if (response.status === "FAILED") {
        stopPolling();
        throw new Error(
          response.error_message || "Video analysis failed on the server.",
        );
      }

      if (response.status === "COMPLETED") {
        stopPolling();
        fetchResults(jobId);
      } else {
        setAnalysisStatus(response.status); // Updates UI to show QUEUED or PROCESSING
      }
    } catch (err) {
      handleError(err);
    }
  };

  const fetchResults = async (jobId) => {
    try {
      setAnalysisStatus("Fetching results...");
      const results = await apiClient.getResults(jobId);
      setResultData(results);
      setPhase(PHASE.COMPLETE);
    } catch (err) {
      handleError(err);
    }
  };

  const handleError = (error) => {
    stopPolling();
    setGlobalError(error.message || "An unexpected error occurred.");
    setPhase(PHASE.ERROR);
  };

  const resetApp = () => {
    stopPolling();
    setCurrentJob(null);
    setResultData(null);
    setGlobalError(null);
    setAnalysisStatus("");
    setPhase(PHASE.IDLE);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <ShieldAlert className="logo-icon" />
        <h1>
          DeepTrace<span className="text-gradient">.ai</span>
        </h1>
      </header>

      <main className="app-main">
        {globalError && (
          <ErrorBanner
            message={globalError}
            onDismiss={() => setGlobalError(null)}
          />
        )}

        {phase === PHASE.IDLE && (
          <VideoUploader onUpload={handleUpload} isLoading={false} />
        )}

        {phase === PHASE.ANALYZING && (
          <AnalysisProgress status={analysisStatus} />
        )}

        {phase === PHASE.COMPLETE && resultData && (
          <ResultsDashboard result={resultData} onReset={resetApp} />
        )}

        {phase === PHASE.ERROR && (
          <div className="panel error-panel">
            <h2>Analysis Failed</h2>
            <p>We encountered an issue processing your video.</p>
            <button
              className="button-primary"
              onClick={resetApp}
              style={{ marginTop: "1rem" }}
            >
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
