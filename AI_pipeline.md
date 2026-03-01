# Full AI Pipeline Architecture & Implementation Details

## 1. Executive Summary
This document provides an in-depth, extremely detailed explanation of the AI pipeline engineered for the Deepfake & AI-Generated Video Detection System. The pipeline is strictly optimized to run on an 8GB NVIDIA RTX 5060 GPU under Fedora Linux. It accurately classifies videos into three distinct classes:
- **Class 0:** Authentic/Real Video
- **Class 1:** AI Face-Swapped Video (Spatial/Deepfake)
- **Class 4:** Fully AI-Generated Video (Sora, Runway ML, etc.)

By migrating away from massive Transformer components (e.g., TimeSformer) to an ultra-efficient `EfficientNet-B4 + FrameLSTM` resident model sequence, we maintain sub-6.5GB VRAM inference usages while achieving highly accurate spatio-temporal detection. The system is designed not just for batch training, but for a synchronous FastAPI live inference architecture that adheres strictly to safe memory allocations without compromising on the deep inductive biases required for generative detection.

---

## 2. Core Architectural Philosophy
Operating within an 8GB VRAM constraint necessitates rigorous memory management and sequential evaluation. Deep learning models, especially visual models tracking spatial dimensions over time, rapidly consume memory bounds. The core pillars of this architecture are:

1. **Resolution Cap:** 
   All frames are evaluated at exactly `224x224`. This prevents quadratic memory explosions common when tuning CNNs at higher resolutions. By keeping the spatial dimensions clamped, we guarantee the feature tensors emitted by the backbone remain consistently sized across all iterations.
   
2. **GPU Residency:** 
   Instead of swapping models between CPU and GPU iteratively (which heavily degrades latency via PCIe bandwidth bottlenecks and encourages memory fragmentation), both `EfficientNet-B4` and `FrameLSTM` remain loaded on the GPU permanently during inference. The device map is explicitly maintained in `.eval()` mode.

3. **In-Flight Memory Trashing:** 
   Memory is immediately freed inside the pipeline. Input arrays and intermediate spatial features are dumped via `del` and a highly specific `torch.cuda.empty_cache()` call guarantees the heap doesn't expand across sliding windows. Python's Garbage Collection is preemptively invoked to ensure tensors drop out of scope before the next iteration is allowed to allocate.

4. **Zero-Disk Slicing:** 
   Inference does not use intermediate PNG/JPG frame dumping. Frame-exact extraction utilizes `ffmpeg` subprocess piping directly to NumPy `uint8` RAM arrays via the `image2pipe` protocol, guaranteeing that flash storage I/O is not a bottleneck.

---

## 3. Data Generation Pipeline (`master.csv`)
Training robust deepfake models requires cleanly structured dataset indices. The `.csv` pipeline is divided into distinct stages to prevent data leakage and provide high-quality bounding box patches.

### Stage 1: Validation (`stage1_validate.py`)
Validates that physical files (Real and Fake videos) exist in the `dataset/raw` directory. 
- Using `ffprobe`, it captures internal container metrics natively.
- Evaluates duration thresholds; specifically ensuring videos are >8 seconds to permit robust temporal feature extraction.
- Excludes corrupted bitstreams before they induce silent tensor failures down the pipeline.
- Outputs `validated_videos.csv` securely tracking the clean subset explicitly.

### Stage 2: Injection Ranges (`stage2_inject.py`)
Calculates synthetic injection bounds for Class 1 (Face Swap) videos. If a video is 30 seconds long, an injection range (e.g., 10s to 18s) is artificially constructed, tracking where the deepfake payload functionally exists. 
- Ensures negative samples (real faces flanking either side of an injection) are categorized accurately, improving the decision boundaries.
- Outputs `injection_log.csv`.

### Stage 3: Sliding Windows & Spatial Bounding (`stage3_window_and_label.py`)
The most computationally heavy dataset step. It strides through every video in 4-second sliding windows (stride of 2 seconds):
- Uses `decord` (with fallback FFmpeg pipelines) to extract exactly 20 frames per window (5 frames-per-second capability).
- **MTCNN Pre-Processing:** Utilizes Multi-Task Cascaded Convolutional Networks to track and draw bounding boxes around human faces in the window. 
- Normalizes these bounding box coordinates to `[0.0 to 1.0]` mapping ratios. This is critical as source videos fluctuate vastly in resolution (720x1280 to 4K). Storing ratios isolates the network from input scale geometries.
- If the current 4-second window temporally intersects the `injection_log.csv` payload bounds, it labels the window with `injection_label = 1`. Otherwise `0`.

### Stage 6: Assured Split (`stage6_csv.py`)
Consolidates the windows into `windows_index.csv`. From there, it safely separates datasets strictly by `video_id`.
- **Leakage Prevention:** A model learning a background wall structure from a train split will inappropriately classify a validation split from the same physical video file, artificially raising accuracy metrics.
- Generating `train.csv`, `val.csv`, and `test.csv` safely guarantees a disjoint distribution of actors and environments.

---

## 4. The PyTorch Data Loader (`dataset/dataloader.py`)
To train without overwhelming the storage drives with millions of physical 224x224 face crops, the `DeepfakeWindowDataset` dynamically slices MP4s on the fly during training:

1. **Initialization:** Receives configuration for scaling, `seq_len` (default 20), `fps` (default 5), and standard transforms (e.g. ColorJitter).
2. **Dynamic Decord Lookup:** Opens the `.mp4` into RAM natively without dumping sequences to disk. Computes the specific chunk sequence mapped by `start_sec` and `end_sec`.
3. **FFmpeg Strict Fallbacks:** If the `--frame_exact` flag is enabled to bypass `decord` keyframe drifts (usually ±1 frame), it spawns an `image2pipe` stream reading literal interleaved RGB structs directly from `stdout`.
4. **Normalized Recovery:** Translates the `[0.0 to 1.0]` normalized bounding box values back into the concrete target resolution (e.g., 1920x1080 -> 4K mappings).
5. **Spatial Clamping:** Clamps the box firmly within array indices `H-1` and `W-1` to stop exception out-of-bounds array slicing errors.
6. **Augmentations:** Resizes faces natively to `224x224`. Translates the image space statically via `torchvision.transforms.Normalize([0.485, 0.456, 0.406])`.
7. **Temporal Seeding:** Captures the `torch` manual seed directly; random spatial augmentations (e.g., horizontally flipping a frame) MUST mirror identically across all 20 frames of the sequence in order to prevent creating synthetic temporal jittering that fools the LSTM logic into identifying the augmentation as a deepfake anomaly.

---

## 5. Network Models and Training Sequence

### Phase 1: Spatial Evaluation - `EfficientNet-B4`
The `scripts/train_efficientnet.py` acts as the massive feature extraction engine fine-tuning a globally pooled EfficientNet.

- **Sequence Inputs:** `[Batch, 20, 3, 224, 224]`.
- **View Transformation:** Operations natively require 4D tensors for 2D Convolutions. The input expands via `x.view(B*T, C, H, W)` into `[Batch*20, 3, 224, 224]`.
- **The Gradients:** Uses `torch.amp.autocast('cuda')` mixed-precision logic executing mathematically intense blocks at `fp16` while storing parameter master weights at `fp32`.
- **Accumulation:** Operates a simulated batch target. Due to the 8GB limit, a batch size of `4` combined with spatial sequences of `20` translates to 80 native EfficientNet passes concurrently. It executes this via loop iterations, summing the gradient backwards, only invoking `optimizer.step()` after executing an exact calculation to hit arbitrary effective batch limits (e.g., 32 effective).
- **Objectives:** This phase focuses on learning intra-frame artifacting:
    - Micro-textures around facial blending seams (where the Face Swap mask adheres to the biological structure).
    - Mismatched chromatic patterns in iris reflections.
    - Missing teeth alignment characteristics inherently unique to AI morphing logic.

### Phase 2: Temporal Evaluation - `FrameLSTM`
Once the EfficientNet reaches optimal spatial competency across `train` splits, its weights are serialized and frozen. `scripts/train_framelstm.py` then takes over:

1. **Frozen Pass:** Executes a strict `torch.no_grad()` evaluation pipeline over the `[Batch*20, 3, 224, 224]` input.
2. **Intermediate Extraction:** Intercepts the tensors explicitly after `global_pool()` natively returning vectors measuring `1792` elements longitudinally. 
3. **Sequence Structuring:** The `[B*T, 1792]` matrix is folded back dynamically into `[Batch, 20, 1792]`.
4. **Recurrent Analysis:** The `LSTM` sequentially cycles through the 20 timesteps carrying over the hidden state matrix to discover chronological coherence flaws.
    - For example, if frame 3 contains a specific light pooling density over the jaw geometry, does frame 4 correspond physically?
    - Many advanced Deepfakes hold up perfectly under single-frame analysis but "shiver" or "micro-pulse" out-of-phase during sequence timelines due to frame-to-frame independence. The FrameLSTM successfully intercepts this anomaly class safely under the 8GB bounds that massive monolithic Transformers typically OOM out on.

---

## 6. Live Inference Pipeline Architecture (FastAPI)

The inference server (`backend/main.py`) acts as the user gateway, executing an optimized synchronous pipeline against a specific MP4 upload.

### Fast Preprocessing Step (`backend/pipeline/preprocessor.py`)
When a random MP4 is uploaded via the `/api/upload` REST endpoint, it skips database lookup. Instead:
- Sweeps structural frames simultaneously.
- Runs MTCNN across consecutive 4-second blocks natively isolated on the heavily multithreaded Central Processing Unit to map coordinates securely without consuming GPU caches statically needed by the neural layers.
- Consolidates median face coordinate smoothing logic and explicitly packs `[20, 224, 224, 3]` arrays directly attached into system DRAM pointers.

### The Resident GPU Sequence (`backend/pipeline/video_branch.py`)
This vital branch handles all calculations while explicitly bypassing PCIe round-trip latency hits.
- Converts arrays natively to `torch.float32`, mapping to ImageNet coordinates and pinning them statically via `.to('cuda')`.
- All blocks are enclosed absolutely inside contextual closures (`with torch.no_grad():` and `with torch.amp.autocast('cuda'):`).
1. **EfficientNet Forward:** Executes extraction emitting `.get_classifier()` metrics indicating spatial swap probabilities, alongside extracting `[1, 20, 1792]` tensor footprints simultaneously.
2. **Direct LSTM Injection:** Bypassing all `.cpu()` and `.detach()` explicit drops, the structural properties merge directly into the resident LSTM state model. The probability floats output cleanly.
3. **Hyper-Aggressive GC:** It intercepts local scope boundaries immediately destroying namespace variables: `del inputs, features, frame_logits, features_seq, temporal_logits`.
4. **Cache Cleared:** A precise `torch.cuda.empty_cache()` flushes any fractional memory fragmentation natively inside the GPU allocator.

### Temporal Aggregator (`backend/pipeline/aggregator.py`)
Deepfake injections are inherently chained temporal segments, not single 4-second flashes. 
- Analyzes overlap logic against the sliding window configurations (4s windows on 2s sliding intervals natively overlap by exactly 50%).
- If Window 0 (0-4s) throws a `0.8` confidence score, and Window 1 (2-6s) throws a `0.82` confidence sequence limit:
- It tracks the gap bridging logic and aggregates them into a singular `"0.0 -> 6.0"` temporal classification label, preventing chaotic flashing overlays across client UI layers.

### Class 4 Generative Heuristics (`backend/pipeline/class4_heuristic.py`)
Sora, Runway, and Luma models intrinsically represent "Fully AI" datasets completely. Because MTCNN often fundamentally fails to securely anchor bounding boxes (due to structural morphing properties that reject facial landmark recognition math logic):
- We measure sequence-level pixel chaos explicitly.
- **Variance Scoring:** Collapses the RGB sequences dynamically onto a mathematically naive `.2989 R` scaling axis forming raw grayscales.
- Computes difference tensors tracking individual voxel morphing across the 20 spatial frames.
- Identifies sequences demonstrating temporal melting sequences failing to adhere to static 3D physical coordinate planes by detecting high global absolute variance measurements.

### Final Classification Decision
Outputs JSON via API encapsulating exactly:
```json
{
    "status": "success",
    "classification": {
        "class_id": 1,
        "label": "AI_VIDEO"
    },
    "flagged_ranges": [
        {"start": 12.0, "end": 26.0, "peak_score": 0.88}
    ]
}
```
Thus completing an edge-hardened classification system strictly confined securely to <8GB physical limits.

---

## 7. Continuous Integration and VRAM Protection (`scripts/ci_*.py`)

A massive risk when deploying generative-detectors on lower-end consumer GPUs (RTX 5060) is code drift inducing Out-Of-Memory (OOM) crashes in production. Even slightly altering a batch parameter or leaving a tensor attached to the computation graph can explode a 6GB footprint into a 12GB footprint instantly.

### 7.1 VRAM Training Failsafe (`cripts/ci_vram_check.py`)
To prevent this, the architecture includes a CI gating mechanism that simulates a single forward and backward pass of the spatial `EfficientNet` at peak unfreezing density.
- Generates synthetic PyTorch tensors natively mapped directly into purely initialized `cuda` memory mapping domains.
- Executes `model(x)` natively under `amp.autocast()`.
- Explicitly queries `torch.cuda.max_memory_allocated()`.
- If the theoretical peak utilization exceeds `7.5 GB` globally, the CI script explicitly executes `sys.exit(1)`, rejecting the commit or configuration change. This mathematically prevents end-users from beginning a 6-hour training run only for the epoch to crash halfway through.

### 7.2 Inference Latency & Peak Profiling (`scripts/ci_infer_check.py`)
Latency is identically catastrophic for real-time validation systems. This script mocks the FastAPI HTTP request wrapper.
- Mounts both `EfficientNet` and `FrameLSTM` purely locally inside the GPU hierarchy.
- Constructs an array mimicking an `ffmpeg` temporal sequence extraction precisely.
- Tests the structural `video_branch.py` evaluation pipeline across 5 sliding window intervals concurrently.
- Enforces a rigorous sub- `6.5 GB` VRAM peak memory assertion metric.
- Mandates a strict < `250 ms` latency bound per window to ensure the real-time playback capability of the frontend UI does not stall while waiting for network responses.

## 8. Conclusion
By strictly clamping spatial resolution (`224x224`), eliminating implicit cross-bus tensor mappings (`.cpu()`), deploying `FrameLSTM` entirely as a GPU-resident temporal validator, and aggressively invoking `empty_cache()` at structural block boundaries, this pipeline achieves state-of-the-art Generative Deepfake Detection capabilities within incredibly bounded hardware environments. It does not rely on disk-caching, avoiding IOPS burnout, and safely streams `ffmpeg` structural blocks strictly via RAM.

