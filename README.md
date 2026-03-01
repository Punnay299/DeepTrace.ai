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

### Prerequisites
* **Python**: 3.11+
* **Node.js**: v18+ (For Frontend UI)
* **GPU**: Minimum 8GB VRAM (NVIDIA CUDA 12+ Required for Inference)
* **FFmpeg**: Must be globally installed on your system path.

### 1. Clone & Initialize Backend
```bash
git clone https://github.com/your-username/DeepTrace.ai.git
cd DeepTrace.ai

# Create Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install Core ML & Backend Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart timm facenet-pytorch opencv-python numpy pandas decord
```

### 2. Download Model Weights
Ensure your pre-trained model weights are positioned correctly at the root directory:
* `/models/efficientnet_b4_best.pth`
* `/models/framelstm_best.pth`

### 3. Initialize Frontend
```bash
cd frontend
npm install
```

---

## 🧪 Testing the Architecture (Dry-Run Verification)

Before starting the server, run the built-in isolated integrity test suite to validate your hardware, PyTorch configurations, database connections, and logic layers:

```bash
python scripts/run_tests.py
```
> *This script synthesizes fake normalized tensors and random data structures to evaluate shape transformations, exception handling, and SQLite queuing boundaries under isolation, requiring absolutely no external video inputs.*

---

## 💻 Running the Application

### Start the FastAPI Backend
From the root project directory:
```bash
python backend/main.py
```
*(Backend runs asynchronously by default to port `8000`, strictly enforcing sequential model execution).*

### Start the React Frontend Dashboard
From the `/frontend` directory:
```bash
npm run dev
```
*(Dashboard opens dynamically to port `5173`).*

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
