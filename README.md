# DeepTrace.ai: Deepfake & AI-Generated Video Detection System

![DeepTrace.ai](https://img.shields.io/badge/Status-Production--Ready-success) ![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012%2B-ee4c2c) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688) ![React](https://img.shields.io/badge/Frontend-React-61dafb)

DeepTrace.ai is a highly optimized, dual-pipeline deepfake detection system engineered to flag both localized AI manipulations (e.g., face-swaps) and fully AI-generated chronological video sequences (e.g., Sora, Runway). 

It is specifically architected to run synchronously within strict **8GB VRAM budgets**, requiring zero disk-trashing during inference, and deploying an asynchronous SQLite queuing mechanism to safely handle concurrent uploads.

---

## 🚀 Key Features

* **3-Class Detection Triage System**:
  * **Class 0 (REAL)**: Authentic video; no AI manipulation or temporal artifacts detected.
  * **Class 1 (AI_VIDEO)**: Partial AI manipulation; detected localized face-swaps or specific injected AI segments via spatial anomalies.
  * **Class 4 (FULLY_AI_VIDEO)**: Fully AI-generated sequence; >85% synthetic coverage or identified via high-variance temporal heuristic indicators (morphing/smearing).
* **Dual-Model Inference Engine**:
  * **EfficientNet-B4**: Extracts raw spatial features per frame to identify pixel-level generative artifacts.
  * **FrameLSTM**: Processes the 1792-dimensional sequences logically across time to uncover temporal irregularities.
* **Aggressive VRAM Protection**: 
  * Strict memory locking (`app.state.inference_lock`) in the backend to ensure sequential PyTorch allocations.
  * Models heavily leverage `torch.amp.autocast('cuda')` (mixed precision).
  * In-flight trashing (`gc.collect()`) physically purges intermediate activation tensors natively.
* **On-the-fly Video Preprocessing**:
  * Reads video frames dynamically directly through `ffmpeg` pipes. No physical disk caching.
  * Automatic `MTCNN` bounds detection and fallback crop resizing explicitly locking sequence inputs to `[1, 20, 3, 224, 224]`.
* **"Flat Dark" React Dashboard**:
  * Zero-dependency API abstraction utilizing native `fetch()` without Axios overhead.
  * Robust, capped polling infrastructure ensuring graceful timeout thresholds and dynamic server error parsing.

---

## 🛠 Tech Stack

### AI Pipeline & Processing
* **PyTorch** & **Torchvision**: Core tensor processing and inference operations.
* **TIMM**: Image Models framework providing the foundational `EfficientNet-B4` architecture.
* **Facenet-PyTorch (`MTCNN`)**: CPU-bound facial recognition and dynamic bounding box mapping.
* **FFmpeg / OpenCV / Decord**: Dynamic temporal decoding and window frame parsing.

### Backend Infrastructure
* **FastAPI**: Synchronous endpoint orchestration (`/upload`, `/status/`, `/result/`).
* **SQLite3**: In-flight job queue management and persistent status tracing across the application lifecycle.
* **Uvicorn**: ASGI web server handling frontend requests natively.

### Frontend Dashboard
* **React**: Component-view synchronization (`App.jsx`, `VideoUploader.jsx`, `ResultsDashboard.jsx`).
* **Vanilla CSS**: Hand-tailored flat dark aesthetic relying entirely on core CSS variables, avoiding Tailwind bloat.

---

## ⚙️ Installation & Setup

> **Note:** This architecture has been **fully statically verified** and its PyTorch spatial-temporal flow is **100% operational** under isolated inference conditions. 

### Prerequisites (Assumed Configured)
* **Python**: 3.11+ environment activated with `requirements.txt` dependencies installed.
* **Node.js**: v18+ environment ready.
* **GPU**: System contains an active NVIDIA GPU with CUDA 12+ capabilities.
* **FFmpeg**: Configured globally.

## 💻 Step-by-Step Execution Guide (From Scratch)

This section outlines how to generate the master dataset, train the AI models from scratch, and launch the web interface.

> **Note:** This assumes your Python (`requirements.txt`) and Node.js environments are already configured. This architecture has been **fully statically verified** and its PyTorch spatial-temporal flow is **100% operational** under isolated inference conditions. 

### Step 1: Supply Raw Video Data
Place your raw `.mp4`, `.webm`, or `.mov` files into the designated raw directories:
* `dataset/raw/class_0_real/`
* `dataset/raw/class_1_ai_video/`
* `dataset/raw/class_4_full_ai_video/`

### Step 2: Generate the Master CSV Pipeline
Run the data generation pipeline sequentially to validate files, inject mock overlaps, generate 4-second overlapped windows with MTCNN bounding boxes, and split the data safely without dumping image frames to disk.

```bash
python dataset/pipeline/stage1_validate.py
python dataset/pipeline/stage2_inject.py --mock
python dataset/pipeline/stage3_window_and_label.py --input_csv dataset/labels/validated_videos.csv --injection_log dataset/labels/injection_log.csv --output_csv dataset/labels/windows_index.csv --device cuda
python dataset/pipeline/stage6_csv.py
```
*(This produces your final `train.csv`, `val.csv`, and `test.csv` in `dataset/labels/`)*

### Step 3: Train the Spatial AI Model (EfficientNet-B4)
Train the spatial backbone to detect pixel-level generative artifacts and face-swap blending seams.
```bash
python scripts/train_efficientnet.py --train_csv dataset/labels/train.csv --val_csv dataset/labels/val.csv
```
*(This automatically outputs the `efficientnet_b4_best.pth` weight file into the `/models` directory).*

### Step 4: Train the Temporal AI Model (FrameLSTM)
Train the recurrent network to evaluate strict chronological consistency, consuming static vectors generated by the frozen EfficientNet.
```bash
python scripts/train_framelstm.py --train_csv dataset/labels/train.csv --val_csv dataset/labels/val.csv --backbone_weights models/efficientnet_b4_best.pth
```
*(This automatically outputs the `framelstm_best.pth` weight file into the `/models` directory).*

### Step 5: Run the Integrity Dry-Test
Before launching the live inference application, verify your hardware, VRAM, and SQLite queue structures execute correctly in isolated dry-run mode:
```bash
python scripts/run_tests.py
```

### Step 6: Boot the FastAPI AI Backend
Open a terminal in the root project directory (`Deepfake.ai/`) and launch the synchronous server. It will lock locally to port `8000`:
```bash
python backend/main.py
```
*(The backend handles the single-thread GPU locking mechanism asynchronously. Keep this terminal open).*

### Step 7: Boot the React UI Dashboard
Open a **second**, separate terminal window, navigate to the frontend directory, install Node dependencies, and launch the development server:
```bash
cd frontend
npm install
npm run dev
```
*(The dashboard will dynamically mount to `http://localhost:5173`. Open this URL in your browser to begin uploading videos).*

---

## 📂 System Architecture Breakdown

* **`dataset/pipeline/`**: The massive data-generation orchestration pipeline, handling validation, face-swap ingestions, bounding box generation (`MTCNN`), labeling methodologies (`stage3_window_and_label.py`), and master CSV generation.
* **`dataset/dataloader.py`**: The dynamic sequence loader dynamically feeding cropped sequence tensors on-the-fly natively to the training sequence algorithms.
* **`scripts/`**: The core AI training mechanics separating the spatial parameter freezing (`train_efficientnet.py`) from the deep-time chronological understanding algorithms (`train_framelstm.py`).
* **`backend/`**: The strict FastAPI API boundaries, handling payload collisions, validation sanitization, database orchestration (`core/database.py`), and the core PyTorch model forward pass logic (`pipeline/video_branch.py`).
* **`frontend/`**: The flat dark user interface dynamically reading endpoints in deterministic interval loops, delivering diagnostic metrics seamlessly in real-time.

---

## 🔍 Validation Protocol

1. Upload `.mp4`, `.webm`, or `.mov` (Under 500MB).
2. The UI switches to `ANALYZING`.
3. The Video Preprocessor (`ffmpeg`/`MTCNN`) strips 4-second temporal sequences.
4. The GPU locks (`inference_lock`).
5. `score_window()` pushes the vector parameters into the combined architecture.
6. The `aggregator` calculates the holistic variance mapping via Luma tracking.
7. Job finishes. The UI maps the JSON structure displaying `Total Coverage`, `Confidence Metrics`, and precise `Flagged Timestamp Ranges`.

---

## ⚖️ License
MIT License. Free for academic, personal, and research applications.
