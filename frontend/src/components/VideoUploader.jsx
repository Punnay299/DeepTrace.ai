import { useState, useRef } from "react";
import { UploadCloud, FileVideo, X } from "lucide-react";
import "./VideoUploader.css";

export function VideoUploader({ onUpload, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [localError, setLocalError] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const validateFile = (file) => {
    setLocalError(null);
    if (!file) return false;

    if (!file.type.startsWith("video/")) {
      setLocalError("Please select a valid video file.");
      return false;
    }

    const maxSizeBytes = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSizeBytes) {
      setLocalError("File exceeds the 500MB limit.");
      return false;
    }

    return true;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  const clearSelection = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setLocalError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="panel uploader-panel">
      <div
        className={`drop-zone ${dragActive ? "active" : ""} ${selectedFile ? "has-file" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !selectedFile && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          onChange={handleChange}
          style={{ display: "none" }}
          disabled={isLoading}
        />

        {selectedFile ? (
          <div className="file-preview">
            <div className="file-info">
              <FileVideo size={32} className="accent-icon" />
              <div>
                <p className="file-name">{selectedFile.name}</p>
                <p className="file-size">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              className="clear-btn"
              onClick={clearSelection}
              disabled={isLoading}
            >
              <X size={20} />
            </button>
          </div>
        ) : (
          <div className="drop-prompt">
            <div className="icon-container">
              <UploadCloud size={48} />
            </div>
            <h3>Upload Video for Analysis</h3>
            <p>Drag & drop your video here, or click to browse</p>
            <span className="format-hint">
              Supports MP4, MOV, WEBM (Max 500MB)
            </span>
          </div>
        )}
      </div>

      {localError && <p className="uploader-error">{localError}</p>}

      <button
        className="button-primary analyze-btn"
        onClick={handleSubmit}
        disabled={!selectedFile || isLoading || localError}
      >
        {isLoading ? "Uploading..." : "Analyze Video"}
      </button>
    </div>
  );
}
