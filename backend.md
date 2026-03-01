# Deepfake & AI-Generated Video Detection System
## Detailed Backend Pipeline Documentation

This document provides a highly detailed, module-by-module breakdown of the FastAPI Backend Pipeline. Refactored into a memory-safe, strictly synchronous architecture, this backend is specifically designed to run on a machine constrained to an 8GB NVIDIA GPU (like an RTX 5060) while serving an automated deep learning analysis pipeline. 

The backend acts as the orchestrator of all AI processing: it safely streams video chunks, evaluates them across spatial and temporal neural network engines, manages atomic database transactions, safeguards GPU VRAM, and aggregates mathematical scoring into human-readable classifications.

---

## 1. Core Entrypoint: `backend/main.py`
The `main.py` module defines the absolute highest logical execution level. It handles exactly three core responsibilities under strict single-concurrency (`uvicorn --workers 1`):

### 1.1 Lifespan Management
FastAPI's `@asynccontextmanager` initiates necessary resources before the server accepts traffic.
- **Database Initializer**: Calls `init_db()` to construct the raw `jobs` and `results` SQLite tables recursively.
- **Model Loader**: Executes the synchronous mounting of both `EfficientNet` and `FrameLSTM` onto the GPU target via `load_all_models(device)`.
- **Protected State**: Bypassing dangerous Python Global Dictionaries, it directly attacks the models to `app.state`, specifically defining `app.state.backbone` and `app.state.lstm`.
- **The Execution Lock**: Crucially instantiates `app.state.inference_lock = threading.Lock()`. Because FastAPI is asynchronous, multiple API requests might overlap GPU inference. This standard Python thread lock ensures that only ONE concurrent process ever accesses `torch.amp.autocast()` simultaneously, definitively preventing OutOfMemory crashes.
- **Teardown**: On server termination, it explicitly dereferences the models utilizing `del app.state.backbone` and triggers `torch.cuda.empty_cache()` to flush the internal VRAM allocator tables cleanly back to the OS.

### 1.2 Routing Registry
It imports `health.py`, `upload.py`, and `status_and_results.py` logic components dynamically integrating them via `app.include_router(router)`. It keeps the main script visually explicit.

---

## 2. The Core Database Layer: `backend/core/database.py`
To provide high-availability tracking of long-running operations (handling video inference takes roughly 1+ minutes per minute of video), the system leverages SQLite explicitly configured for async web-workers.

### 2.1 Connection Integrity
- Uses `sqlite3.connect(..., check_same_thread=False)` which is mathematically required for FastAPI. Wait timeouts are configured to `10.0s`.
- Sets `PRAGMA foreign_keys = ON` natively protecting database relationship corruption (preventing results being stored against non-existent jobs).

### 2.2 Table Schemas
- **jobs**: Tracks the active execution pipeline state. Attributes: `job_id`, `status` (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`), and `error_message`.
- **results**: A secondary connected schema strictly mapped against successful jobs. Houses the complete formatted JSON classification output mapping directly back to the API.

### 2.3 Atomic Handlers
Every function handles `with conn:` transactions internally mapping implicit BEGIN and COMMIT protocols ensuring thread safety:
- `create_job(job_id)`: Attempts raw SQLite `INSERT OR IGNORE`.
- `update_job_status()`: Specifically writes states immediately bypassing Python memory structures.
- `save_result_and_complete()`: Bundles an INSERT to `results` and the `COMPLETED` flag update into a single unified locked database write.

---

## 3. High-Security Upload Orchestration: `backend/api/upload.py`
The `/api/upload` endpoint represents the most complex execution path logic in the entire network. Being blocked strictly by the `app.state.inference_lock`, it acts as the primary controller.

### 3.1 Input Validation Sandbox (`validate_video`)
Before touching the database or triggering GPU operations, the input is defensively attacked:
- Validates the byte bounds natively against a 500MB upload hard limit.
- Executes `ffprobe` in a strict blocking subprocess utilizing a `10s` timeout parsing embedded container metadata.
- Rejects inputs lacking explicit `"codec_type": "video"`.
- Prevents OOM bounding attacks by confirming the height/width ratios fall mathematically under `1920x1080`.
- Implements FPS protections, failing processing if extraction logic will burst above 60 FPS natively.
- Enforces a 60-second limit specifically to prevent extremely long batch timeouts.

### 3.2 Streaming Execution & AI Generation Loop
Once validated, the controller transitions the job state to `PROCESSING`. Wrapped strictly inside `with lock:`, the following loop runs sequentially:
1. Calls `stream_crops(...)` mapping the FFmpeg Python generator natively yielding exactly 20-frame increments.
2. Checks an explicit internal watchdog timer: if `time.time() - start_time > 120`, it triggers a localized `TimeoutError` intentionally breaking the process gracefully rather than freezing the host server.
3. Transmits the frame tensors to `score_window(...)`.
4. Computes temporal weighting rules `(0.4 x spatial) + (0.6 x temporal)`.
5. Iteratively appends data to `window_results`.
6. Passes the raw generated window lumas specifically to the `class4_heuristic` scoring sequence.

### 3.3 The Three-Class Triage Algorithm
Once processing concludes, the controller routes classification logically mapped from overlapping precedence limits explicitly designed in the `deepfake.md` documentation:
- Calculates `flagged_ranges` boundaries against an aggressive 0.55 confidence score bound.
- Identifies `video_coverage` mapping the percentage of time identified against total length mathematically.
- Rules logic flow:
  1. *Rule A (Maximum Triage)*: If percentage >= 85%, assign `4: FULLY_AI_VIDEO`
  2. *Rule B (Explicit Feature)*: If explicit seams identified (ranges > 0), assign `1: AI_VIDEO`. Explicit features ALWAYS trump fuzzy variance metrics.
  3. *Rule C (Fuzzy Variance)*: If No ranges BUT heuristics variance >= 0.65 threshold, assign `4: FULLY_AI_VIDEO`.
  4. *Rule D (Default Safety)*: Assumed `0: REAL`.

Finally, the controller wraps the logic natively inside `save_result_and_complete` flushing tracking logic to disk arrays, clearing VRAM internally explicitly and finalizing the HTTP POST Return.

---

## 4. Polling Mechanisms: `backend/api/status_and_results.py`
Because the upload endpoint is a massive blocking operation taking significant server time, a frontend client requires instantaneous updates avoiding HTTP gateway timeouts.

- **`GET /api/status/{job_id}`**: Fast path database read probing the `jobs` SQLite layer looking for the QUEUED, PROCESSING, COMPLETED, or FAILED flags. Returns explicit `error_message` strings allowing frontend UIs to visually map validation failures seamlessly.
- **`GET /api/result/{job_id}`**: Retrieves the JSON blob successfully saved to `results` when the processing logic returned successful. Blocks HTTP 400 structures defensively if requested before `status == 'COMPLETED'`.

---

## 5. Visual Intelligence Subsystems

### 5.1 Memory Streaming Extractor: `backend/pipeline/preprocessor.py`
Originally designed as an OOM-crashing array builder throwing entire MP4 sequences locally into `16GB+` RAM structs, it operates entirely as a Python Generator `yield` sequence explicitly limiting VRAM structures.
- Parses metadata for exact native resolution sizes dynamically using explicit substring configurations.
- Executes `ffmpeg -f image2pipe ...` attaching `stdout` buffers dynamically extracting raw rgb24 structs dynamically.
- Tracks `frame_idx` dynamically filling a buffer variable internally. Once the 20-frame window buffer is saturated, it halts ffmpeg piping sequences dynamically natively.
- **MTCNN Subprocessing**: Sweeps bounding boxes recursively targeting human tracking configurations dynamically pulling crops securely over `cv2.resize`.
- If NO face identifies internally over the 20 frame subset, the system safely falls back pulling a `224x224` center scaled geometry preserving system architecture inputs dynamically. 
- Discharges the array immediately via `yield` destroying memory overhead allocations internally.

### 5.2 GPU Evaluator Array: `backend/pipeline/video_branch.py`
Receives the yielded 20-frame `[T,C,H,W]` window natively.
- Converts to `float32` and normalizes using standardized ImageNet distributions efficiently natively mapped mathematically directly onto `cuda` memory mapping targets.
- Wraps executions absolutely inside `with torch.no_grad():` and `with torch.amp.autocast('cuda'):`
- Extracts high dimensionality vectors sequentially pushing bounds inside the frozen EfficientNet weights cleanly returning `[20, 1792]` vector mappings dynamically.
- Immediately slices intermediate memory bounds natively passing into `FrameLSTM` internally scoring `temporal_logit` metrics intelligently assessing LSTM logic mathematically.
- Triggers hard Python explicit GC configurations: `del inputs, features, frame_logits, features_seq, temporal_logits` and natively sweeps `gc.collect()`.

### 5.3 Frame Agility Tracker: `backend/pipeline/class4_heuristic.py`
Determines fully manipulated videos cleanly mapping structural variances over entire datasets natively.
- Computes `0.2989 R, 0.5870 G, 0.1140 B` exact luma bounds.
- Runs purely across numpy CPU arrays sequentially generating frame-to-frame `diff` mappings dynamically returning numerical variance averages predicting `FULLY_AI` sequence limits dynamically vs naturally stable human physics.

### 5.4 Sequence Aggregator: `backend/pipeline/aggregator.py`
Eliminates user interface flickering identifying exact bounds recursively mapping 4-second blocks spanning into logical blocks natively bridging overlaps bridging tolerance `gap <= tolerance_sec` explicitly merging bounding structures dynamically.

---

## 6. Structural Integration Validation (`scripts/`)
Deploying deep learning networks under hardware constraints necessitates absolute defensive integration scripts ensuring configuration files don't accidentally overload internal mappings natively. 
- **`ci_vram_check.py`**: A CI build script initializing backward bounds recursively mapping memory limits explicitly proving the `EfficientNet-B4` accumulator footprint successfully terminates mathematically underneath `< 4.7 GB` natively clearing the 8GB baseline limit cleanly.
- **`ci_infer_check.py`**: Explicitly maps pure spatial-temporal generation sequentially natively blocking execution directly asserting inference runs safely operating under optimal `750ms` sequence boundaries ensuring frontend processing operates cleanly against video extraction logic blocks seamlessly mapping boundaries defensively.
