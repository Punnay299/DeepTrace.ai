# Deepfake & AI-Generated Video Detection System
## Complete Frontend Architecture Documentation

This document provides a comprehensive, highly detailed, module-by-module breakdown of the React-based Frontend Application. Purpose-built to interface flawlessly with the synchronous, hardware-constrained FastAPI backend, this frontend emphasizes extreme stability over ephemeral flashiness. 

The architecture entirely rejects heavyweight frontend frameworks (e.g., Next.js, TailwindCSS) and external dependency chains (e.g., Axios), strictly relying on Vanilla CSS and native browser Fetch capabilities. The user interface leverages a "Flat Dark UI" aesthetic optimized for maximum analytical clarity.

---

## 1. Core Architectural Philosophy

Building a UI for generative AI detection involves unique UX challenges. Specifically, AI inference utilizing complex Dual-Model approaches (EfficientNet-B4 + FrameLSTM) on an 8GB NVIDIA RTX 5060 requires substantial processing time (often >60 seconds per video). The frontend must orchestrate a psychologically reassuring user journey that masks this latency without resorting to brittle WebSockets.

### 1.1 The "Flat Dark" Aesthetic
The UI deliberately avoids trendy, performance-degrading CSS rules such as `backdrop-filter: blur()`. Glassmorphism causes rendering lag on lower-end client machines when stacked atop HTML5 Video processing limits.
- **Base Canvas:** The void (`#0f0f0f`) establishes a cinema-grade backdrop.
- **Elevation Panels:** Structured logical bounds (`#1a1a1a`, `#222222`) visually partition distinct data phases.
- **Singular Cybernetic Accent:** Electric Cyan (`#00f5ff`) denotes active processing, user engagement, and primary focal paths natively preventing visual noise.

### 1.2 Dependency Minimization
- **HTTP Transport:** Replaced `axios` entirely with a native, robust `fetch` wrapper. Eliminating third-party networking dependencies drastically reduces the Vite bundle footprint and shields against dependency-chain vulnerabilities.
- **CSS Strategy:** Employs pure Global CSS Variables (`--colors`, `--radii`) within `index.css`. This prevents the "class soup" inherent to Tailwind and allows immediate overriding of themes.

---

## 2. Global State Orchestration: `src/App.jsx`

At the center of the frontend hierarchy acts `App.jsx`, functioning as a strict finite state machine governing user interaction seamlessly.

### 2.1 The Phase Graph
The UI mathematically restricts user workflow into exactly four encapsulated phases (`PHASE` enumeration):
1. **`IDLE`**: Displays the Drag & Drop upload handler. Awaits valid binary payloads.
2. **`ANALYZING`**: Triggers upon HTTP 200 from `/api/upload`. Instantiates a mathematically capped background polling loop protecting against infinite server hangs.
3. **`COMPLETE`**: Engages immediately when `/api/status/{job_id}` returns the `COMPLETED` flag. Transitions completely to the `ResultsDashboard`.
4. **`ERROR`**: A terminal failure route. Engages when backend validation fails, network connectivity drops, or background analytical loops crash internally.

### 2.2 The Asynchronous Polling Subsystem
FastAPI runs synchronously against a Threading Lock securing GPU constraints. Consequently, the primary HTTP POST `/api/upload` is intrinsically asynchronous to backend execution. 

To bridge this gap cleanly, `App.jsx` executes a specialized `setInterval`:
- **Interval Width:** `POLL_INTERVAL_MS = 2500` (2.5 seconds). This natively prevents DDOSing the local backend with micro-queries while maintaining near-instant UX feedback loops.
- **The Hard Cap Protocol:** `pollCountRef.current > 60`. If the backend fails to terminate analysis within 150 seconds (~2.5 minutes), the frontend forcibly self-terminates the loop, abandoning the job, and routing the UI immediately into an explicit timeout Error Banner. This guarantees the browser never locks infinitely.

### 2.3 Garbage Collection & Polling Safeties
- `useEffect(() => { return () => stopPolling(); }, [])`
- `clearInterval(pollTimerRef.current)`
- If the user hastily navigates away or the main root component unmounts unexpectedly, the background interval timer is annihilated explicitly, preventing memory leaks natively in the V8 engine.

---

## 3. The API Abstraction Layer: `src/api/client.js`

To consolidate HTTP logic globally, `client.js` wraps the native browser Fetch API constructing a unified gateway pointing explicitly at `http://localhost:8000/api`.

### 3.1 Custom Error Sub-Typing
Because FastAPI utilizes explicit `HTTPException(status_code=400, detail="reason")` bounds during upload validation constraints (e.g. file size overages, invalid codec detections), the UI must parse these cleanly.
- `client.js` defines an ES6 `class ApiError extends Error`.
- When `response.ok` evaluates to boolean `false`, the helper dynamically parses the JSON payload, grabs the `data.detail` or `data.error_message` variable explicitly, wraps it inside an `ApiError`, and throws it directly back up the lexical scope into the `try/catch` block of `App.jsx`.

### 3.2 The Execution Pointers
- **`uploadVideo(file)`**: Instantiates a native JavaScript `FormData()` object, appends the binary `File` explicitly, and initiates the overarching POST vector.
- **`checkStatus(jobId)`**: The lightweight GET probing mechanism checking database flags passively. 
- **`getResults(jobId)`**: The heavy GET extraction endpoint capturing the finalized 3-class result tree.

---

## 4. Input Gateway: `src/components/VideoUploader.jsx`

The primary data acquisition terminal natively blocks corrupted payloads before executing network traversal.

### 4.1 Visual Bounding
- Encapsulated inside a dashed border zone (`.drop-zone`).
- Translates dynamic CSS classes on `onDragEnter` converting the zone from empty placeholder into a scaled `var(--accent-cyan-dim)` glowing acceptor.

### 4.2 Hard Validation Rules (`validateFile`)
Replicating backend validation checks exactly locally saves immense network latency transferring doomed files.
1. **Type Constraint:** Explicitly blocks anything lacking a `.startsWith('video/')` MIME type descriptor. No PDFs. No JPGs.
2. **Payload Weighting:** Establishes `maxSizeBytes = 500 * 1024 * 1024` explicitly converting mathematically bounding uploads under 500MB seamlessly preventing the `uvicorn` backend worker from executing an OOM dump upon buffer saturation.
3. If validation fails, it routes the reason directly to a localized red-text sub-string instead of destroying the global application state. 

### 4.3 Input Ref Hiding
The physical `<input type="file">` HTML tag is highly un-style-able cross-browser (Chrome vs Firefox). The component utilizes React's `useRef(null)` hooking mechanism to hide the element completely using `display: none` and triggering programmatic clicks exclusively. 

---

## 5. Temporal Latency Visualizer: `src/components/AnalysisProgress.jsx`

As the backend iteratively parses through 4-second overlapped chunks evaluating spatial frames through `EfficientNet-B4` and temporal sequences through `FrameLSTM`, the user stares at this loading panel.

### 5.1 Dual-Tier Animation Sequencing
Static spinners imply frozen processes. The `AnalysisProgress` component cascades:
- **Central Focus:** A `lucide-react` `<Loader2>` SVG bound internally to a strict linear `spin` 2.0s keyframe duration.
- **Pulse Wrapper:** A secondary wrapping container fading opacity dynamically over `pulse` keyframes mathematically proving to the user the browser loop remains completely attached to the execution state.
- **The Bounding Bar:** To mask indeterminate progress tracking (since the frontend does not know the exact frame count of the video uploaded implicitly without breaking encapsulation boundaries), an internal `.progress-bar-indeterminate` slider loops natively left-to-right via `calc` bounding.

### 5.2 Server State Syncing
The `status` prop translates the raw backend SQL tracking string into human-friendly syntax:
- `QUEUED` -> *"Waiting in queue..."*
- `PROCESSING` -> *"Analyzing video frames..."*
Providing visceral proof the request didn't merely stall out in HTTP space.

---

## 6. Deterministic Output: `src/components/ResultsDashboard.jsx`

The final rendering engine. Converts the mathematical tensors dumped by FastAPI out of `app.state` into visceral color-coded logic. 

### 6.1 The Triage Switch Case
Translates the structural output integer classes (`0`, `1`, `4`) cleanly mathematically:
- **Class 0 `real`:** The authentic baseline. Triggers `.status-real` CSS (`#00ff88` bright green). 
- **Class 1 `ai`:** The localized injection. Triggers `.status-ai` (`#ffb800` amber warning). The model successfully calculated that while the majority of the MP4 is mathematically real, specific ranges present temporal logic flaws suggesting an InsightFace or generic faceswap payload internally.
- **Class 4 `full-ai`:** The Generative AI limit. Triggers `.status-full-ai` (`#ff3366` pure crimson danger). Identifies absolute Sora/Runway generational bounds mathematically failing heuristic structural checks explicitly.

### 6.2 The Plain Text Timestamp Paradigm
Instead of rendering a convoluted visual scrubber or pseudo-video player, the design mathematically isolates timestamps natively preventing UX confusion.
- Renders an explicit `<ul>` iteration over `flagged_ranges`. Each item dynamically executes local mathematical formatting `<div className="time-range">` converting raw seconds (`125.4`) into standardized `M:SS` mapping (`02:05`).
- Displays the peak confidence logic explicitly proving to the user the thresholding bounds (`0.55`) successfully executed seamlessly. 

### 6.3 Aggregated Metric Blocks
Outputs three explicit raw-data variables: `Heuristic Variance` (tracking Luma divergence tracking), `Flagged Coverage` % bounding, and integer `Windows Processed` tracking directly proving the dual-model pipeline successfully evaluated the entire timeline explicitly bypassing early termination.

---

## 7. Global Exception Catching: `src/components/ErrorBanner.jsx`

When the network protocol crashes violently or the backend Python layer traps an internal unhandled exception, `ErrorBanner` takes over perfectly.
- Translates `throw new Error("xyz")` directly out of the polling timeout function seamlessly natively inside a top-mounted `index.css` slide-down animation sequence.
- Renders highly explicit string errors directly back at the user natively preventing silent fail loops inherently common within asynchronous single-page-applications.

---

## 8. Vanilla CSS Scaling: `src/index.css`

A deeply scoped breakdown of the global styling rules applied seamlessly bypassing external preprocessors.

### 8.1 The Variables Table
- `:root` declares physical hexadecimal alignments logically eliminating inline mapping internally explicitly controlling structural components mathematically preventing stray visual artifacts natively separating themes seamlessly.

### 8.2 Fluid Typography
- Pulls `Outfit` natively via Google Web Fonts dynamically integrating geometric sans-serif properties logically bounding headers heavily cleanly preventing variable height shifting dynamically.
- Native `border-box` sizing recursively prevents bounding box overflow calculations crashing layout flow dynamically. 

### 8.3 Central Layout Constraint
- `.app-container` establishes an explicit `max-width: 1000px;` bounding limit ensuring ultra-wide monitors do not stretch the Uploader panel into an unwieldy letterbox shape natively centering cleanly using `margin: 0 auto;`

---

## Conclusion
The frontend layer perfectly mimics the engineering priorities of the deep-learning backend: strictly typed, zero-configuration dependency logic, mathematical bounded execution tracking, forced garbage collection, and precise explicit textual error handling preventing UI hallucination explicitly across highly computational latency tracks.
