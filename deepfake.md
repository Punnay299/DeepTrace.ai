# Deepfake & AI-Generated Video Detection System
## Complete System Design Documentation — Version 3.0 (Video-Only, Hackathon-Ready)

---

> **What changed from v2.0 → v3.0**
> Audio detection (Classes 2, 3, 5, 6) removed entirely. System is now video-only.
> 7-class system reduced to 3 operative classes.
> Hackathon backend (sync FastAPI + SQLite) documented as primary build target.
> Full production backend kept as Section 15 (post-hackathon roadmap).
> VRAM management, Fedora 43 setup, training strategy, debugging guide, and
> known error catalog added from real conversation analysis.
> All 20 fixes from v2.0 retained where applicable.

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Classification System — 3 Operative Classes](#2-classification-system--3-operative-classes)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Data Strategy & Dataset Design](#4-data-strategy--dataset-design)
5. [Data Pipeline — Raw Videos to Labeled CSVs](#5-data-pipeline--raw-videos-to-labeled-csvs)
6. [AI Model Architecture](#6-ai-model-architecture)
7. [Detection Pipeline](#7-detection-pipeline)
8. [Class 4 Heuristic — Fully AI Video Detection](#8-class-4-heuristic--fully-ai-video-detection)
9. [Backend System Design — Hackathon Build](#9-backend-system-design--hackathon-build)
10. [Frontend ↔ Backend ↔ AI Connection](#10-frontend--backend--ai-connection)
11. [VRAM Management on 8GB GPU](#11-vram-management-on-8gb-gpu)
12. [Training Strategy & Checkpointing](#12-training-strategy--checkpointing)
13. [Fedora 43 Environment Setup](#13-fedora-43-environment-setup)
14. [Known Errors, Debugging Guide & Fixes](#14-known-errors-debugging-guide--fixes)
15. [Configuration & Environment Variables](#15-configuration--environment-variables)
16. [Limitations & Honest Expectations](#16-limitations--honest-expectations)
17. [Post-Hackathon Production Roadmap](#17-post-hackathon-production-roadmap)

---

# 1. Project Overview

This system is a **video-only AI-powered deepfake detection platform**. Users upload videos (30 seconds to 5 minutes long) and the system analyzes video frames to detect AI manipulation, returning a classified verdict with precise timestamps showing exactly which parts of the video are suspicious.

Audio detection (voice cloning, synthetic speech) has been deliberately excluded from v3.0 to make the system buildable in 5 days on a single 8GB VRAM laptop. Audio detection is documented in Section 17 as the first post-hackathon extension.

### What The System Detects (v3.0 Scope)

| Modality | What it looks for |
|---|---|
| **Video** | AI-generated faces, face swaps, fully AI-generated scenes, unnatural temporal motion, GAN/diffusion artifacts in face regions |

### Core Design Philosophy

```
NOT: "Is this video real or fake?" (binary — too simple)
BUT: "What was manipulated and exactly when?" (precise + useful)
```

### Hardware Target

```
GPU:  NVIDIA RTX 5060 (8GB VRAM)
OS:   Fedora 43 Linux
RAM:  System RAM (laptop, assumed 16GB+)
```

---

# 2. Classification System — 3 Operative Classes

The system outputs one of 3 classes, each with associated timestamps where relevant.
The label indices 0, 1, 4 are preserved from the 7-class design so the API schema
is forward-compatible when audio classes are added post-hackathon.

| Index | Label | Description | Timestamps Returned? |
|---|---|---|---|
| 0 | **REAL** | Entire video is authentic. No AI manipulation detected. | No |
| 1 | **AI_VIDEO** | AI-generated or face-swapped segments detected in specific parts | Yes — video timeline |
| 4 | **FULLY_AI_VIDEO** | Entire video is AI-generated (Sora, RunwayML, Pika, Kling, SVD, etc.) | No (full coverage) |

> **Why indices 0, 1, 4?**
> The full 7-class system uses indices 0–6. Classes 2, 3, 5, 6 are audio-related and
> excluded from this build. Preserving the original indices means the database schema,
> API responses, and frontend code require zero changes when audio is added later.
> Never renumber them to 0, 1, 2 — that creates a migration problem.

### Class Distinction Logic

```
FULL vs PARTIAL determination:

flagged_duration ÷ total_duration >= 85%  →  FULLY_AI_VIDEO (Class 4)
flagged_duration ÷ total_duration <  85%  →  AI_VIDEO (Class 1)
Nothing flagged                           →  REAL (Class 0)
```

### Classification Decision Tree

```
Start
  │
  ├── video_coverage >= 85%
  │   └── Class 4: FULLY_AI_VIDEO
  │       Note: EfficientNet on FaceForensics++ may UNDERDETECT Class 4
  │       (Sora/Runway have different artifacts than faceswaps)
  │       → Class 4 heuristic assists here (see Section 8)
  │
  ├── 0% < video_coverage < 85%
  │   └── Class 1: AI_VIDEO (with timestamps)
  │
  └── video_coverage == 0%
      └── Class 0: REAL
```

### Why Class 4 Is The Hardest Problem

```
Class 1 (AI_VIDEO) detection:
  EfficientNet-B4 fine-tuned on FaceForensics++ is GOOD at this.
  Face-swap artifacts, blending seams, GAN fingerprints are its specialty.
  Pretrained weights transfer well. Strong signal.

Class 4 (FULLY_AI_VIDEO) detection:
  Sora, Runway, Pika, Kling generate entire videos with NO face-swap seam.
  FaceForensics++ models were never trained on these artifacts.
  A fully AI-generated video of a dog running will score LOW on the faceswap detector.
  This is why Section 8 (heuristic layer) is non-optional for Class 4.
```

---

# 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER (Browser/App)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP (HTTPS in production)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI APP                                   │
│                         (Synchronous — SQLite Backend Worker)           │
│                                                                         │
│   POST /api/upload   GET /api/status/{id}   GET /api/result/{id}       │
│                                                                         │
│   Middleware: File Type → Codec Check → Size → Duration Check          │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────┐   ┌──────────────────────────────────────────────┐
│   FILE STORAGE   │   │                  SQLITE DB                    │
│                  │   │  • Job tracking (status, progress)            │
│  /tmp/uploads/   │   │  • Results (verdict, timestamps, windows)    │
│  /tmp/frames/    │   │  • Single file, zero config                  │
└──────────────────┘   └──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI PIPELINE                                   │
│                                                                         │
│  ┌─────────────┐   ┌──────────────────────────────────────────────┐   │
│  │ PREPROCESSOR│   │              VIDEO BRANCH                     │   │
│  │             │   │                                               │   │
│  │ FFmpeg:     │──►│  MTCNN face crop @224×224                    │   │
│  │ • frames    │   │  EfficientNet-B4 (batched [40,3,224,224])    │   │
│  │   @10fps    │   │         +                                    │   │
│  │   @224×224  │   │  FrameLSTM (8-frame subsample)             │   │
│  └─────────────┘   │  [FALLBACK: FrameLSTM if FrameLSTM OOMs]  │   │
│                    └────────────────┬─────────────────────────────┘   │
│                                     │                                  │
│                    ┌────────────────▼─────────────────────────────┐   │
│                    │         CLASS 4 HEURISTIC LAYER               │   │
│                    │  Calculate sequence luma variations            │   │
│                    │  (assists FULLY_AI_VIDEO detection)           │   │
│                    └────────────────┬─────────────────────────────┘   │
│                                     │                                  │
│                    ┌────────────────▼─────────────────────────────┐   │
│                    │         TEMPORAL AGGREGATOR                   │   │
│                    │  • Score per window                           │   │
│                    │  • Merge contiguous flagged ranges            │   │
│                    │  • Compute coverage %                         │   │
│                    └────────────────┬─────────────────────────────┘   │
│                                     │                                  │
│                    ┌────────────────▼─────────────────────────────┐   │
│                    │         DECISION ENGINE                       │   │
│                    │  3-class output + timestamps + confidence     │   │
│                    └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                          SQLite DB
                  (job status + results stored)
```

---

# 4. Data Strategy & Dataset Design

## 4.1 Minimum Viable Dataset for Hackathon

```
For a working demo you do NOT need 20GB of data.
You need enough to fine-tune EfficientNet-B4 and validate it.

Hackathon target:
  Total Dataset Size:  ~4–6 GB
  Videos per class:    50–100 videos minimum
  Class count:         3 classes only

Class 0 (Real):            50–100 real videos
Class 1 (AI_VIDEO):        50–100 real videos WITH face-swap injection
Class 4 (FULLY_AI_VIDEO):  50–100 fully AI-generated videos

This is enough to demonstrate the system working correctly.
Accuracy will improve with more data, but 50 per class is sufficient to demo.
```

## 4.2 Per-Class Data Strategy

| Class | Min Videos | Source Strategy | Injection Needed? |
|---|---|---|---|
| Class 0 — Real | 50–100 | Download real videos (YouTube CC, Pexels, news footage) | No |
| Class 1 — AI Video | 50–100 | Real videos + synthetic face-swap injected at random timestamps | Yes |
| Class 4 — Fully AI Video | 50–100 | Download from Sora demos, RunwayML, Pika Labs, Kling AI, Stable Video Diffusion | No |

## 4.3 Where to Get Data Fast

### Class 0 — Real Videos
```
Sources (all free, no scraping needed):
  Pexels.com         → royalty-free videos, direct MP4 download
  Pixabay.com        → same, large library
  YouTube CC license → download with yt-dlp (pip install yt-dlp)
  
  Command to download 720p MP4 with yt-dlp:
    yt-dlp -f "bestvideo[height<=720][ext=mp4]+bestaudio/best" --merge-output-format mp4 <URL>

Time required: 2–4 hours to collect 50–100 clips
```

### Class 1 — Face-Swap Injection
```
Tool: InsightFace (NOT DeepFaceLab — too slow for 5 days)

  InsightFace's inswapper_128.onnx model:
    • Single pass face swap per frame
    • Much faster than DeepFaceLab
    • Good enough quality to produce detectable artifacts

  pip install insightface onnxruntime-gpu

  Injection process:
    1. Pick a random timestamp range (see Section 5.3 for formula)
    2. Extract frames in that range
    3. Run insightface inswapper on those frames
    4. Re-splice with FFmpeg back into original video
    5. Record start/end in injection_log.csv

  Alternative if InsightFace setup is slow: SimSwap
    github.com/neuralchen/SimSwap (has pretrained models, single command)

Time for injection: ~30–60 seconds per video (GPU)
For 50 videos: ~1 hour total GPU time
```

### Class 4 — Fully AI Generated Videos
```
Sources:
  RunwayML Gen-3   → runwayml.com (free trial credits, ~10 videos)
  Pika Labs        → pika.art (free tier, generate short clips)
  Kling AI         → klingai.com (free tier available)
  Stable Video Diff → run locally (huggingface.co/stabilityai/stable-video-diffusion)
  Sora demos       → openai.com/sora (pre-released sample videos, publicly available)
  
  Existing repositories of AI-generated videos:
    github.com/grip-unina/DMimageDetection (research datasets)
    FaceForensics++ (has DeepFakeDetection subset — request access)
    DFDC dataset (Facebook, publicly available, has fully generated videos)

IMPORTANT: Download variety — different tools leave different artifacts.
           Having only Runway videos makes Class 4 tool-specific.
           Aim for: 20 Runway + 20 Pika + 10 SVD minimum.

Time required: 2–3 hours to generate/download 50 clips
```

## 4.4 Why Synthetic Injection for Class 1 (Not Downloaded Deepfakes)

```
Problem with downloading existing deepfakes:
  ✗ Unknown manipulation timestamps — you don't know when the fake starts
  ✗ Need manual human labeling (slow, error-prone)
  ✗ Human labeling introduces bias
  ✗ Cannot verify exact start/end of manipulation for timestamp training
  ✗ Limited supply of timestamp-labeled real-world deepfakes

Solution — Synthetic generation:
  ✓ YOU create the manipulation → YOU know exact timestamps
  ✓ Labels are mathematically perfect (written at injection time)
  ✓ Zero human labeling bias
  ✓ Full control over difficulty, duration, position
  ✓ Reproducible
```

## 4.5 Video Length Distribution

```
For hackathon:
  Aim for a mix of lengths per class:
    30%  × 30–60 second videos   (fast to process, good for debugging)
    40%  × 1–2 minute videos
    30%  × 2–5 minute videos

  Do not use only short videos — model will overfit to short-video temporal patterns.
  Do not use only 5-minute videos — injection and frame extraction takes much longer.
```

## 4.6 Domain Gap — The Hidden Risk

Even with perfect synthetic data, a gap exists between training data and real-world fakes:

```
Your Training Data:               Real-World Deepfakes:
━━━━━━━━━━━━━━━━━━━               ━━━━━━━━━━━━━━━━━━━━
Clean 1080p source    vs.         Downloaded from internet
Direct tool output    vs.         Post-processed to hide artifacts
No compression        vs.         Multiple recompression cycles

Gap = Model trained on clean fakes may miss messy real-world fakes
```

### Augmentation to Bridge Domain Gap

Apply these augmentations during training (NOT to raw files — apply in the DataLoader):

| Augmentation | Range | Purpose |
|---|---|---|
| Random JPEG compression | Quality 40–85% | Simulate reupload/re-encoding |
| Random Gaussian noise | σ = 0.01–0.05 | Camera sensor noise |
| Random brightness | Factor 0.8–1.2 | Lighting variation |
| Random blur | Kernel 1–3px | Re-encoding blur |
| Random horizontal flip | 50% probability | Augment face crop diversity |

```python
# In your PyTorch Dataset __getitem__:
import torchvision.transforms as T
import random

augment = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomGrayscale(p=0.05),   # rare — prevents color-only artifact detection
])

# Apply only during training, not validation/test
if self.split == 'train':
    frame = augment(frame)
```

## 4.7 Training Data Distribution (Windows)

```
With 50 videos per class, mixed lengths averaging ~90 seconds each:

Class 0 (Real):           ~1,500–2,500 windows
Class 1 (AI Video):       ~1,300–2,000 windows (only fake windows are labeled 1)
Class 4 (Fully AI):       ~1,500–2,500 windows (all windows labeled 1)
─────────────────────────────────────────────────
Total:                    ~4,300–7,000 windows

This is enough to fine-tune EfficientNet-B4 from ImageNet pretrained weights.
It is NOT enough to train from scratch (needs 10x more).
NEVER train from scratch on this dataset size.
```

---

# 5. Data Pipeline — Raw Videos to Labeled CSVs

## 5.1 Pipeline Overview

```
RAW VIDEOS (organized in class folders)
            │
            ▼
┌──────────────────────────────────────┐
│  STAGE 1: VALIDATION                 │
│  Verify format, duration, resolution │
│  Verify codec is supported           │
│  OUTPUT: validated_videos.csv        │
└──────────────────┬───────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
     Class 1            Classes 0, 4
  (needs injection)    (pass-through)
          │                 │
          ▼                 │
┌─────────────────────┐     │
│  STAGE 2: INJECTION │     │
│  InsightFace swap   │     │
│  at random timestamps     │
│  WRITE injection log│     │
│  OUTPUT:            │     │
│  injection_log.csv  │     │
│  ★ SOURCE OF TRUTH  │     │
└──────────┬──────────┘     │
           │                │
           └────────┬───────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  STAGE 3: WINDOWING                  │
│  4s window / 2s stride               │
│  MTCNN face crop per frame @224×224  │
│  Extract frames @10fps per window    │
│  OUTPUT: windows_index.csv           │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  STAGE 4: LABELING                   │
│  JOIN windows + injection log        │
│  Apply overlap rule for Class 1      │
│  Hardcode labels for Classes 0, 4   │
│  OUTPUT: window_labels.csv           │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  STAGE 5: AUGMENTATION               │
│  Applied in DataLoader (NOT on disk) │
│  Labels completely unchanged         │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  STAGE 6: CSV GENERATION             │
│  Split at VIDEO level (NEVER window) │
│  train.csv 80%                       │
│  val.csv   10%                       │
│  test.csv  10%                       │
└──────────────────────────────────────┘
```

## 5.2 Folder Structure

```
dataset/
├── raw/
│   ├── class_0_real/
│   │   └── (mp4 files, any length 30s–5min)
│   ├── class_1_ai_video/
│   │   └── (real mp4 files — injection applied in Stage 2)
│   └── class_4_full_ai_video/
│       └── (AI-generated mp4 files downloaded directly)
│
├── synthetic_assets/
│   └── swap_faces/          ← source face images for InsightFace injection
│
├── processed/
│   └── windows/
│       └── frames/          ← extracted face-cropped frame sets per window (224×224)
│
├── labels/
│   ├── validated_videos.csv
│   ├── injection_log.csv    ← THE SINGLE SOURCE OF TRUTH for Class 1 timestamps
│   ├── windows_index.csv
│   ├── window_labels.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
└── pipeline/
    ├── stage1_validate.py
    ├── stage2_inject.py
    ├── stage3_window.py
    ├── stage4_label.py
    └── stage6_csv.py
```

## 5.3 Stage 1 — Validation

```
INPUT:  All raw video files from class folders
OUTPUT: validated_videos.csv

Checks per video:
  ├── File format is mp4/mov/avi/mkv
  ├── Duration: 30 seconds minimum, 300 seconds maximum
  ├── Minimum resolution: 360p height
  └── Codec is in [h264, h265, vp8, vp9, av1, mpeg4]

Use ffprobe for all checks — do not rely on file extension:
  ffprobe -v quiet -print_format json -show_streams -show_format <file>

validated_videos.csv schema:
  video_id | class | path | duration | height | codec | status | error
```

## 5.4 Stage 2 — Injection (Class 1 Only)

### Injection Configuration

```
MIN_INJECT_DURATION  = 8.0 seconds  (minimum length of face-swapped segment)
MAX_INJECT_FRACTION  = 0.60         (inject at most 60% of video duration)
MIN_GAP_FROM_EDGE    = 5.0 seconds  (never inject in first/last 5 seconds)

Why 5s gap from edges:
  Prevents edge artifacts from being confused with injection artifacts
  Ensures clean "before" and "after" context for the model to learn from
```

### Random Timestamp Generation

```python
import random

def generate_inject_range(duration: float) -> tuple[float, float]:
    max_inject_duration = min(duration * 0.60, duration - 10.0)
    if max_inject_duration < 8.0:
        raise ValueError(f"Video too short to inject: {duration}s")
    inject_duration = random.uniform(8.0, max_inject_duration)
    max_start = duration - inject_duration - 5.0
    inject_start = random.uniform(5.0, max_start)
    inject_end = inject_start + inject_duration
    return round(inject_start, 2), round(inject_end, 2)

# Example for 120s video:
# max_inject_duration = min(72, 110) = 72s
# inject_duration = random(8, 72) → e.g. 31.4s
# inject_start = random(5, 83.6) → e.g. 23.4s
# inject_end = 54.8s
```

### Injection Process with InsightFace

```python
# Rough pseudocode — implement in stage2_inject.py
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 = GPU 0

swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)

def inject_faceswap(video_path, inject_start, inject_end, source_face_img, output_path):
    # 1. Extract frames in inject range with FFmpeg
    # 2. Run face detection + swap per frame
    # 3. Re-encode swapped frames back into video with FFmpeg
    # 4. Splice: original[0:inject_start] + swapped + original[inject_end:]
    pass
```

### Injection Log — Source of Truth

```
injection_log.csv — written IMMEDIATELY when injection happens
Never recalculate timestamps. Always read from this file.

Columns:
  video_id        | UUID of processed video
  class           | always 1 for this pipeline
  duration        | total video duration in seconds
  v_inject_start  | start of face-swap injection
  v_inject_end    | end of face-swap injection

Example row:
  abc-123 | 1 | 120.5 | 23.4 | 54.8
```

## 5.5 Non-Injection Classes (0, 4) — Pass-Through

```
Class 0 (Real):
  Videos are authentic → pass directly to Stage 3
  No injection log entry created

Class 4 (Fully AI Video):
  Videos are entirely AI-generated → pass directly to Stage 3
  No injection log entry created
```

## 5.6 Stage 3 — Windowing

```
Configuration:
  WINDOW_SIZE_SEC    = 4.0 seconds
  STRIDE_SEC         = 2.0 seconds  (50% overlap)
  FRAME_SAMPLE_RATE  = 10 fps

  FRAMES_PER_WINDOW  = 4.0 × 10 = 40 frames

Window count per video duration:
  30 seconds → ~14 windows
  1 min      → ~29 windows
  2 min      → ~59 windows
  3 min      → ~89 windows
  5 min      → ~149 windows

No cap applied — ALL windows used for training.
Removing the cap ensures training distribution matches inference distribution.
A 5-minute video contributes 149 windows, not a capped 30.

FACE CROP per Frame (FIX from v2.0):
  Before storing each frame, run MTCNN face detection.
  If face detected  → crop to face bounding box, resize to 224×224
  If no face found  → use full frame resized to 224×224 as fallback
  This ensures EfficientNet-B4 focuses on face regions rather than background.
  face_detected=True/False recorded per window for transparency.

  Library: facenet-pytorch
    pip install facenet-pytorch

OUTPUT: windows_index.csv
  video_id | class | win_start | win_end | frame_dir | face_detected
```

## 5.7 Stage 4 — Labeling

### The Overlap Rule (Class 1)

```
For a window [win_start, win_end] and injection range [inject_start, inject_end]:

  IS FAKE = (win_end > inject_start) AND (win_start < inject_end)

  This catches:
  ├── Windows fully inside injection range
  ├── Windows partially overlapping at the START of injection
  └── Windows partially overlapping at the END of injection

Example:
  Injection range: 23.4s ──────────────────── 54.8s

  Window [20s-24s]: 24 > 23.4 AND 20 < 54.8  → TRUE  → video_fake=1
  Window [22s-26s]: 26 > 23.4 AND 22 < 54.8  → TRUE  → video_fake=1
  Window [52s-56s]: 56 > 23.4 AND 52 < 54.8  → TRUE  → video_fake=1
  Window [55s-59s]: 59 > 23.4 AND 55 < 54.8  → FALSE → video_fake=0
```

### Hardcoded Labels (Classes 0, 4)

```
Class 0: ALL windows → video_fake=0
Class 4: ALL windows → video_fake=1

No overlap calculation needed — labels are deterministic by class definition.
```

### Synchronization Guarantee

```
KEY RULE: Stage 4 NEVER recalculates timestamps.
          It ONLY reads from injection_log.csv.

  Stage 2: Generates timestamp → writes to injection_log.csv IMMEDIATELY
  Stage 3: Generates windows   → writes to windows_index.csv
  Stage 4: Reads BOTH files    → applies overlap rule

  injection_log.csv is the single contract between Stage 2 and Stage 4.
  Desynchronization is mathematically impossible with this approach.
```

## 5.8 Stage 6 — CSV Generation (No Stage 5 on Disk)

```
Augmentation is applied IN the PyTorch DataLoader, NOT by writing augmented files to disk.
Writing augmented files to disk wastes storage and time.
Apply transforms in __getitem__ during training only.

Split at VIDEO level (NEVER at window level):

WRONG: Split individual windows into train/val/test
  → Same video's windows appear in both train and test
  → Data leakage → artificially inflated accuracy numbers

CORRECT: Split entire videos into train/val/test
  → ALL windows from video abc-123 go to train
  → ALL windows from video def-456 go to test
  → Zero data leakage possible

Split ratios:
  train.csv: 80% of videos (and all their windows)
  val.csv:   10% of videos
  test.csv:  10% of videos

master.csv schema:
  video_id | class | win_start | win_end | video_fake | frame_dir | face_detected | split
```

## 5.9 Complete Label Matrix

```
         │ Needs    │ Stage 4    │ video_fake
Class    │ Stage 2? │ Logic      │ per window
─────────┼──────────┼────────────┼────────────
Class 0  │ NO       │ Hardcode   │ always 0
Class 1  │ YES      │ Overlap    │ 0 or 1
Class 4  │ NO       │ Hardcode   │ always 1
```

---

# 6. AI Model Architecture

## 6.1 Overview — Why Two Models

```
Single model cannot handle all detection types:
  • Face/video artifacts require SPATIAL understanding (single frame analysis)
  • Motion artifacts require TEMPORAL understanding (sequence of frames)

Solution: Two specialized models, each expert in one domain,
          combined through a weighted scoring system.

Video-only pipeline removes audio models entirely.
No Wav2Vec2. No Mel CNN. No VAD. No audio extraction.
```

## 6.2 Model 1 — EfficientNet-B4 (Frame-Level CNN)

```
Purpose: Analyze individual face crops for spatial AI artifacts

What it detects:
  ├── Unnatural skin texture (GAN fingerprints)
  ├── Blurry/inconsistent hair edges (blending seam artifacts)
  ├── Eye artifacts (asymmetry, reflection inconsistency)
  ├── GAN frequency-domain fingerprints in pixel patterns
  └── Lighting inconsistencies on face region

FIX (from v2.0) — Input resolution is 224×224:
  EfficientNet-B4's native resolution is 224×224, NOT 224×224.
  Using 224×224 defeats the accuracy advantage of B4 over smaller variants.
  All preprocessing must resize frames/crops to 224×224.

FIX (from v2.0) — MTCNN face crop before EfficientNet:
  Every frame goes through MTCNN face detection first.
  EfficientNet sees a cropped face at 224×224, not the full scene.
  This prevents the model from learning background artifacts instead of face artifacts.
  If no face detected → pass full frame resized to 224×224 (fallback).

Architecture:
  Input:  Single face-cropped frame [3, 224, 224] RGB
  Model:  EfficientNet-B4 pretrained on ImageNet (timm library)
  Head:   Replace final classifier → Linear(num_features, 1)
  Output: Single score 0.0 (real) to 1.0 (fake)
  Loss:   BCEWithLogitsLoss

FIX (from v2.0) — Batched forward pass (critical for speed):
  WRONG (sequential — 40 GPU calls per window):
    scores = [model(frame.unsqueeze(0)) for frame in frames]

  CORRECT (batched — 1 GPU call per window):
    batch = torch.stack(frames)   # shape: [40, 3, 224, 224]
    scores = model(batch)         # shape: [40, 1]
    frame_mean = scores.mean()    # single aggregated score per window

  This reduces GPU overhead by 40× per window. Not optional.

Fine-tuning strategy (Phase 1 — frozen backbone):
  1. Load pretrained EfficientNet-B4 weights from timm:
       model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
  2. Freeze ALL existing layers
  3. Add classification head: nn.Linear(model.num_features, 1)
  4. Train ONLY the head for 5 epochs (lr=1e-3)
  5. Validation loss plateau → proceed to Phase 2

Fine-tuning strategy (Phase 2 — partial unfreeze):
  1. Unfreeze last 3 blocks of EfficientNet-B4
  2. Train with lr=1e-5 for 5 more epochs
  3. Use lr scheduler: ReduceLROnPlateau(patience=2, factor=0.5)
  4. Save checkpoint after every epoch (non-negotiable)

Why EfficientNet-B4 specifically:
  B0 is too small — misses subtle artifacts
  B7 is too large — will OOM on 8GB VRAM during training
  B4 is the sweet spot: accuracy vs VRAM at 224×224 resolution

VRAM usage during training:
  Batch size 32 at [32, 3, 224, 224]:  ~4.5–5.5GB
  Leaves ~2.5–3.5GB for MTCNN + system
  If OOM: reduce batch size to 16 → ~2.5–3GB
```

## 6.3 Model 2 — FrameLSTM (Temporal Model)

```
Purpose: Analyze sequences of frames for temporal artifacts

What it detects:
  ├── Flickering between frames (AI video instability)
  ├── Unnatural physics (hair, clothes, background elements)
  ├── Inconsistent lighting changes across time
  ├── Temporal coherence failures in AI-generated video (too smooth or too jerky)
  └── Motion artifacts that single frames cannot reveal

FIX (from v2.0) — Frame count corrected from 40 to 8:
  facebook/FrameLSTM-base-finetuned-k400 was pretrained on 8 frames per clip.
  Passing 40 frames causes shape mismatch and produces garbage output or crashes.

  Subsampling strategy:
    frames_for_FrameLSTM = frames[::5]   # take every 5th from 40 → 8 frames
    This uniformly samples the 4-second window for temporal coverage.
    EfficientNet still receives all 40 frames.
    The subsampling is ONE LINE of code with zero architectural changes.

Architecture:
  Input:  Sequence of 8 frames [1, 8, 3, 224, 224]
          (subsampled from 40 frames — every 5th frame)
          Note: FrameLSTM uses 224×224, not 224×224 — different from EfficientNet
  Model:  facebook/FrameLSTM-base-finetuned-k400 (HuggingFace)
  Head:   Replace classifier → Linear(768, 1)
  Output: Single score 0.0 (real) to 1.0 (fake)

VRAM warning on 8GB:
  FrameLSTM at inference:  ~2.5–3.5GB
  EfficientNet at inference: ~1.5–2GB
  Both simultaneously:       ~4–5.5GB  → might be fine
  Both during training:      ~6–7.5GB  → risky, likely OOM

  SOLUTION: Run EfficientNet and FrameLSTM SEQUENTIALLY during inference.
  Not in parallel. After each model call:
    torch.cuda.empty_cache()
  See Section 11 for full VRAM management strategy.

Fine-tuning strategy:
  1. Load facebook/FrameLSTM-base-finetuned-k400
  2. Freeze all transformer blocks
  3. Replace and train classification head only (5 epochs, lr=1e-3)
  4. Optionally unfreeze last 2 blocks (5 more epochs, lr=1e-5)
  NOTE: Fine-tuning FrameLSTM on 8GB may OOM. See fallback below.

FALLBACK if FrameLSTM OOMs during TRAINING:
  Replace FrameLSTM with FrameLSTM:
    → Extract EfficientNet penultimate layer features per frame (768-dim vector)
    → Stack 40 feature vectors: [40, 768]
    → Feed through LSTM(input_size=768, hidden_size=256, num_layers=2)
    → Linear head: Linear(256, 1)
  This is much lighter (fits easily in 4GB), still learns temporal patterns.
  Accuracy will be somewhat lower but training becomes trivial.
  Decision: if FrameLSTM training OOMs after 2 attempts → switch to FrameLSTM.
```

## 6.4 Video Branch Score Combination

```
frame_scores   = EfficientNet(batch_of_40_frames)  → mean of 40 scores (1 GPU call)
temporal_score = FrameLSTM(8_subsampled_frames)  → single score

video_score = (0.4 × frame_mean) + (0.6 × temporal_score)

Why 60% weight to FrameLSTM:
  Temporal artifacts are harder to fake and more reliable indicators.
  Spatial artifacts alone can produce false positives on stylized real content.

If using FrameLSTM fallback (instead of FrameLSTM):
  frame_features = EfficientNet_penultimate(batch_of_40_frames)  → [40, 768]
  temporal_score = FrameLSTM(frame_features)
  video_score = (0.4 × frame_mean) + (0.6 × temporal_score)
  Weights unchanged — FrameLSTM replaces FrameLSTM in the formula.
```

## 6.5 Overall Confidence — Definition

```
FIX (from v2.0) — overall_confidence is explicitly defined:

For Class 1 (AI_VIDEO) or Class 4 (FULLY_AI_VIDEO):
  overall_confidence = peak_video_score among all flagged windows
  (highest single window score = strongest evidence of manipulation)

For Class 0 (REAL):
  overall_confidence = 1.0 - mean(all_window_video_scores)
  (confidence it is authentic = inverse of average fake probability)

Why max(peak) over average:
  Average is diluted by clean windows in a partially-fake video.
  The peak represents the strongest evidence found.
  Users care about the worst detected moment, not the average.

overall_confidence stored as FLOAT 0.0–1.0
returned in API response as "confidence"
```

## 6.6 Pre-trained Model Sources

| Model | Library | Pretrained On | HuggingFace/PyPI |
|---|---|---|---|
| EfficientNet-B4 | `timm` | ImageNet | `timm.create_model('efficientnet_b4', pretrained=True)` |
| FrameLSTM | `transformers` | Kinetics-400 | `facebook/FrameLSTM-base-finetuned-k400` |
| MTCNN | `facenet-pytorch` | VGGFace2 | `from facenet_pytorch import MTCNN` |

---

# 7. Detection Pipeline

## 7.1 Sliding Window Configuration

```
WINDOW_SIZE_SEC   = 4.0 seconds
STRIDE_SEC        = 2.0 seconds  (50% overlap)
FRAME_SAMPLE_RATE = 10 fps
FRAMES_PER_WINDOW = 40 frames

Why 4s window / 2s stride:
  • Face manipulation needs 2–4s to catch blink/micro-expression artifacts
  • AI video temporal artifacts emerge over 3–5s
  • 50% overlap ensures no artifact falls entirely at a window boundary
  • At 2s stride, a 4s fake segment is caught by at least 2 windows

Windows generated per video length:
  30 seconds → ~14 windows
  1 minute   → ~29 windows
  3 minutes  → ~89 windows
  5 minutes  → ~149 windows
```

## 7.2 Per-Window Scoring

```python
# pseudocode for video_branch.py

import torch
import torch.nn.functional as F

def analyze_window(window_frames: list, start_sec: float, end_sec: float, 
                   backbone: torch.nn.Module, lstm: torch.nn.Module, device: torch.device) -> dict:
    """
    Analyzes a single window of parsed frames using both AI models.
    """
    
    # 1. Prepare Tensors
    # window_frames contains [frames_per_window, height, width, channels] arrays
    tensors = [torch.from_numpy(f).float().permute(2, 0, 1) / 255.0 for f in window_frames]
    batch = torch.stack(tensors).to(device)  # Shape: [N, 3, 224, 224]
    
    # 2. Extract Spatial Features via EfficientNet
    with torch.no_grad():
        spatial_features = backbone.forward_features(batch) 
        spatial_pool = backbone.global_pool(spatial_features) 
        if backbone.drop_rate > 0.:
            spatial_pool = F.dropout(spatial_pool, p=backbone.drop_rate, training=False)
        spatial_logits = backbone.classifier(spatial_pool)
        
        spatial_probs = torch.sigmoid(spatial_logits)
        spatial_score = spatial_probs.mean().item()
        
    # Free memory between models
    del spatial_features, spatial_probs
    torch.cuda.empty_cache()

    # 3. Extract Temporal Features via FrameLSTM
    # Subsample to 8 frames as expected by the LSTM
    stride = max(1, len(window_frames) // 8)
    frames_for_lstm = batch[::stride][:8].unsqueeze(0)  # Shape: [1, 8, 3, 224, 224]
    
    with torch.no_grad():
        lstm_logits = lstm(pixel_values=frames_for_lstm).logits
        temporal_score = torch.sigmoid(lstm_logits).item()

    # Free memory explicitly
    del batch, frames_for_lstm
    torch.cuda.empty_cache()

    # 4. Integrate Model Scores 
    # Weighted calculation
    score = (0.4 * spatial_score) + (0.6 * temporal_score)

    return {
        "start": start_sec,
        "end": end_sec,
        "score": score,
        "spatial": spatial_score,
        "temporal": temporal_score
    }
```

## 7.3 Temporal Aggregation

### Step 1 — Flag Windows

```
FLAG_THRESHOLD = 0.60

For each window:
  flagged = video_score >= 0.60

Tune this on your validation set.
Small changes have large effects:
  0.50 → more false positives, fewer missed fakes
  0.70 → fewer false positives, more missed fakes
  0.60 → balanced starting point
```

### Step 2 — Merge Contiguous Flagged Windows into Ranges

```
GAP_TOLERANCE_SEC   = 2.0 seconds
MIN_RANGE_DURATION  = 1.5 seconds

Algorithm:
  1. Collect all flagged windows
  2. Sort by start time
  3. If gap between consecutive flagged windows < 2.0s → merge them
     (avoids choppy output like [10-14s, 15-18s] when it should be [10-18s])
  4. Discard merged ranges shorter than 1.5s (noise filter)
  5. Record avg_confidence and peak_confidence per range

Example:
  Flagged windows: [10-14s], [12-16s], [14-18s], [30-34s], [32-36s]
  Gap between 18s and 30s = 12s > 2.0s tolerance → separate ranges

  Result: Range 1: [10s–18s] avg=0.83 peak=0.91
          Range 2: [30s–36s] avg=0.76 peak=0.81
```

```python
def merge_flagged_windows(flagged_windows: list[dict],
                          gap_tolerance: float = 2.0,
                          min_duration: float = 1.5) -> list[dict]:
    if not flagged_windows:
        return []

    sorted_wins = sorted(flagged_windows, key=lambda w: w['start'])
    ranges = []
    current = {
        'start': sorted_wins[0]['start'],
        'end': sorted_wins[0]['end'],
        'scores': [sorted_wins[0]['video_score']]
    }

    for win in sorted_wins[1:]:
        if win['start'] - current['end'] <= gap_tolerance:
            current['end'] = max(current['end'], win['end'])
            current['scores'].append(win['video_score'])
        else:
            if (current['end'] - current['start']) >= min_duration:
                ranges.append({
                    'start': current['start'],
                    'end': current['end'],
                    'avg_confidence': sum(current['scores']) / len(current['scores']),
                    'peak_confidence': max(current['scores'])
                })
            current = {'start': win['start'], 'end': win['end'], 'scores': [win['video_score']]}

    if (current['end'] - current['start']) >= min_duration:
        ranges.append({
            'start': current['start'],
            'end': current['end'],
            'avg_confidence': sum(current['scores']) / len(current['scores']),
            'peak_confidence': max(current['scores'])
        })

    return ranges
```

### Step 3 — Compute Coverage Percentage

```python
def compute_coverage(ranges: list[dict], total_duration: float) -> float:
    if not ranges or total_duration == 0:
        return 0.0
    flagged_seconds = sum(r['end'] - r['start'] for r in ranges)
    return min(flagged_seconds / total_duration, 1.0)

FULL_COVERAGE_THRESHOLD = 0.85

video_full = video_coverage >= 0.85
```

## 7.4 Classification Decision Engine

```python
def classify(
    video_ranges: list[dict],
    video_coverage: float,
    total_duration: float,
    class4_heuristic_score: float  # from Section 8
) -> dict:
    """
    3-class decision engine for video-only pipeline.
    """

    FULL_COVERAGE_THRESHOLD = 0.85
    CLASS4_HEURISTIC_THRESHOLD = 0.65   # tune on val set

    video_full    = video_coverage >= FULL_COVERAGE_THRESHOLD
    video_partial = 0 < video_coverage < FULL_COVERAGE_THRESHOLD

    # Priority order matters — check most specific first

    # Case 1: Video fully covered by AI flag
    if video_full:
        return {
            "label": "FULLY_AI_VIDEO",
            "label_index": 4,
            "confidence": compute_confidence(video_ranges, 'full'),
            "video_ranges": [],  # full coverage — no timestamps needed
            "description": "Entire video appears to be AI-generated"
        }

    # Case 2: Partial video flag
    if video_partial:
        return {
            "label": "AI_VIDEO",
            "label_index": 1,
            "confidence": compute_confidence(video_ranges, 'partial'),
            "video_ranges": video_ranges,
            "description": "AI-generated or manipulated content detected at specific timestamps"
        }

    # Case 3: Low model score BUT heuristic suggests fully AI generated
    # (handles Sora/Runway videos that EfficientNet misses — see Section 8)
    if class4_heuristic_score >= CLASS4_HEURISTIC_THRESHOLD:
        return {
            "label": "FULLY_AI_VIDEO",
            "label_index": 4,
            "confidence": class4_heuristic_score,
            "video_ranges": [],
            "description": "Video detected as fully AI-generated via temporal consistency analysis",
            "detection_method": "heuristic"
        }

    # Case 4: Nothing detected
    return {
        "label": "REAL",
        "label_index": 0,
        "confidence": 1.0 - (sum(w['video_score'] for w in video_ranges) / max(len(video_ranges), 1)),
        "video_ranges": [],
        "description": "No AI manipulation detected"
    }
```

## 7.5 Threshold Tuning Guide

These four numbers control detection behavior more than the models:

| Threshold | Default | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `FLAG_THRESHOLD` | 0.60 | Fewer windows flagged (fewer false positives) | More windows flagged (fewer missed fakes) |
| `FULL_COVERAGE_THRESHOLD` | 0.85 | Harder to classify as "fully AI" | Easier to classify as "fully AI" |
| `GAP_TOLERANCE_SEC` | 2.0s | Larger merged ranges | More fragmented ranges |
| `MIN_RANGE_DURATION_SEC` | 1.5s | More noise filtered out | Short artifacts reported |
| `CLASS4_HEURISTIC_THRESHOLD` | 0.65 | Fewer fully-AI false positives | Catches more generative video |

**Always tune on validation set, never on test set.**

---

# 8. Class 4 Heuristic — Fully AI Video Detection

This section addresses the critical gap: EfficientNet trained on FaceForensics++ does not reliably detect fully AI-generated videos (Sora, Runway, Pika) because those videos have no face-swap seam artifacts. A separate heuristic layer is required.

## 8.1 Why EfficientNet Alone Fails on Class 4

```
FaceForensics++ trains the model to look for:
  • Face-swap blending seams
  • Inconsistent skin texture boundaries
  • GAN fingerprints at the merge zone

Sora/Runway/Pika videos have:
  • No face-swap seam (the face was GENERATED, not swapped)
  • Consistent but artificially smooth texture throughout
  • Perfect temporal consistency (too perfect for a real camera)
  • No natural camera shake or sensor noise
  • Unnaturally stable lighting

EfficientNet looking for a seam it doesn't find → scores LOW → classified as REAL
This is the Class 4 detection gap.
```

## 8.2 The Heuristic: Temporal Consistency Score

```
Key insight: Real camera videos have natural variance.
             Fully AI-generated videos are unnaturally consistent.

Two signals used:

SIGNAL 1 — Frame-to-Frame Difference Variance:
  Real video:    frame differences are noisy (camera shake, motion blur, sensor noise)
  AI video:      frame differences are smooth and consistent

SIGNAL 2 — Optical Flow Magnitude Variance:
  Real video:    optical flow has natural irregular patterns
  AI video:      optical flow is too smooth, too uniform

Combined into CLASS4_HEURISTIC_SCORE = 0.0 (real-like) to 1.0 (AI-like)
```

## 8.3 Implementation

```python
import numpy as np
from PIL import Image

def calculate_heuristic_score(frames: list[Image.Image]) -> float:
    """
    Computes a heuristic score (0.0 to 1.0) indicating how 'AI-generated' a video window might be,
    focusing on Class 4 (FULLY_AI_VIDEO) detection. It analyzes temporal structural variance.
    """
    if len(frames) < 2:
        return 0.0

    # 1. Convert to Luma (Y) channel to focus on structural/lighting variance, ignoring color shifts.
    # PIL's convert('L') uses the transform: L = R * 299/1000 + G * 587/1000 + B * 114/1000
    luma_frames = [np.array(frame.convert('L'), dtype=np.float32) for frame in frames]

    # 2. Calculate frame-to-frame pixel differences
    diff_means = []
    for i in range(1, len(luma_frames)):
        diff = np.abs(luma_frames[i] - luma_frames[i-1])
        diff_means.append(np.mean(diff))
    
    # 3. Calculate Variance of Differences
    # Real videos (handheld, natural motion) have high variance in differences.
    # AI generated sequences (Sora, Runway) often interpolate too smoothly, leading to low variance.
    variance = np.var(diff_means) if len(diff_means) > 1 else 0.0

    # 4. Scoring Logic (Inverse Logic)
    # High variance -> Real (score approaches 0.0)
    # Low variance -> Fully AI Generated (score approaches 1.0)
    
    # Threshold Constants (Requires tuning on validation set)
    # If variance >= REAL_TYPICAL, score = 0.0
    # If variance <= AI_TYPICAL, score = 1.0
    VAR_REAL_TYPICAL = 150.0  
    VAR_AI_TYPICAL = 20.0     

    if variance >= VAR_REAL_TYPICAL:
        heuristic_score = 0.0
    elif variance <= VAR_AI_TYPICAL:
        heuristic_score = 1.0
    else:
        # Linear interpolation between thresholds
        heuristic_score = 1.0 - ((variance - VAR_AI_TYPICAL) / (VAR_REAL_TYPICAL - VAR_AI_TYPICAL))

    return float(np.clip(heuristic_score, 0.0, 1.0))
```

## 8.4 Integration into Pipeline

```python
# In your main inference pipeline (analyze.py):

# 1. Run per-window scoring (Section 7.2)
window_scores = [score_window(frames, models) for frames in all_windows]

# 2. Aggregate and get video_coverage (Section 7.3)
flagged_windows = [w for w in window_scores if w['video_score'] >= FLAG_THRESHOLD]
video_ranges    = merge_flagged_windows(flagged_windows)
video_coverage  = compute_coverage(video_ranges, total_duration)

# 3. Run Class 4 heuristic on full video frames (once, not per window)
all_frames_sample = sample_frames_evenly(all_video_frames, n=100)  # don't use all 9000
class4_score = compute_class4_heuristic(all_frames_sample, total_windows=len(window_scores))

# 4. Classify
result = classify(video_ranges, video_coverage, total_duration, class4_score)
```

## 8.5 Limitations of the Heuristic

```
What it catches well:
  ✓ Sora, Runway, Pika videos (too-smooth motion)
  ✓ Stable Diffusion video outputs (uniform texture)
  ✓ Any fully generated video with limited natural camera noise

What it may mis-classify as AI:
  ✗ Screen recordings (inherently smooth — no camera)
  ✗ Animated content / cartoons
  ✗ Heavily stabilized camera footage
  ✗ Timelapse videos

What it may miss:
  ✗ Future AI video tools that intentionally add camera shake noise
  ✗ AI video composited with real footage background

Disclosure in UI:
  When detection_method == "heuristic" in the API response,
  the frontend should display a note:
  "Detected via motion consistency analysis — may be less accurate on
   screen recordings and animated content."
```

---

# 9. Backend System Design — Hackathon Build

## 9.1 Architecture Decision: Sync FastAPI + SQLite

```
Production target (post-hackathon):
  NGINX → FastAPI → Celery → Redis → PostgreSQL

Hackathon target (what actually ships in 5 days):
  FastAPI (synchronous) → SQLite

Rationale:
  ├── Celery adds 3–4 hours of setup, debugging, serialization issues
  ├── Redis adds another service to manage and debug
  ├── PostgreSQL setup and schema migration adds time
  ├── For a single-user demo: synchronous processing is fine
  ├── SQLite: zero config, single file, all Python standard library
  └── A demo waiting 30 seconds beats a demo that crashes during judging

Upgrade path: Section 17 documents the production migration.
              The code structure is designed to make this migration straightforward.
```

## 9.2 Project Folder Structure

```
backend/
│
├── main.py                      ← FastAPI app, startup, CORS, model loading
├── requirements.txt
│
├── api/
│   ├── upload.py                ← POST /api/upload
│   ├── status_and_results.py    ← GET  /api/status/{job_id}, GET /api/result/{job_id}
│   └── health.py                ← GET  /api/health
│
├── core/
│   └── database.py              ← SQLite connection + table creation
│
└── pipeline/
    ├── preprocessor.py          ← FFmpeg streaming frame extraction
    ├── video_branch.py          ← EfficientNet (batched) + FrameLSTM (8-frame)
    ├── aggregator.py            ← Window aggregation + range merging
    ├── class4_heuristic.py      ← Temporal consistency analysis
    └── model_loader.py          ← Load models ONCE at startup
```

## 9.3 Database Schema (SQLite)

```sql
-- database.py — run at startup

CREATE TABLE IF NOT EXISTS jobs (
    job_id               TEXT    PRIMARY KEY,
    status               TEXT    NOT NULL DEFAULT 'QUEUED',
    error_message        TEXT
);

CREATE TABLE IF NOT EXISTS results (
    result_id            TEXT    PRIMARY KEY,
    job_id               TEXT    NOT NULL REFERENCES jobs(job_id),
    result_json          TEXT    NOT NULL
);
```

## 9.4 Model Loading — Critical Pattern

```python
# main.py — load models ONCE at startup, not per request

from contextlib import asynccontextmanager
from fastapi import FastAPI
import torch
import threading
from backend.pipeline.model_loader import load_all_models
from backend.core.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load all models into GPU memory
    init_db()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, lstm = load_all_models(device)
    
    # Store directly on app.state to bypass dangerous global dict mutations
    app.state.backbone = backbone
    app.state.lstm = lstm
    app.state.device = device
    app.state.inference_lock = threading.Lock()
    yield
    # Shutdown: cleanup
    if hasattr(app.state, 'backbone'):
        del app.state.backbone
    if hasattr(app.state, 'lstm'):
        del app.state.lstm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

```

## 9.5 API Endpoints — Complete Specification

### POST /api/upload

```
Request:
  Content-Type: multipart/form-data
  Body:         file (video binary)

Validations (in order):
  1. File type: extension must be mp4/mov/avi/mkv         → 400 if invalid
  2. File size: max 500MB                                  → 400 if too large
  3. Video duration: 30s–300s (check with ffprobe)        → 400 if out of range
  4. Codec: h264/h265/vp8/vp9/av1/mpeg4                   → 400 if unsupported

Processing (synchronous — waits until complete):
  1. Save file to /tmp/uploads/{job_id}.ext
  2. Create job record in SQLite (status=QUEUED)
  3. Run full AI pipeline (blocks until done)
     - Updates job status to PROCESSING at start
     - Updates progress periodically
  4. Save result to SQLite
  5. Update job status to COMPLETED
  6. Delete temp files
  7. Return 200 with result

Response 200 (after processing complete):
  {
    "status": "success",
    "classification": { "class_id": 1, "label": "AI_VIDEO" },
    "flagged_ranges": [...],
    "heuristic_score": 0.45,
    "diagnostics": { "total_windows_processed": 14 }
  }

Error responses:
  400: { "error": "Invalid file type. Accepted: mp4, mov, avi, mkv" }
  400: { "error": "File too large. Maximum: 500MB" }
  400: { "error": "Video must be 30 seconds to 5 minutes" }
  400: { "error": "Unsupported codec" }
  500: { "error": "Internal server error", "detail": "..." }
```

### GET /api/status/{job_id}

```
Response 200:
  {
    "job_id": "abc-123",
    "status": "PROCESSING",
    "error_message": null
  }

Status values:
  QUEUED      → Job created, not started yet
  PROCESSING  → Pipeline running
  COMPLETED   → Fetch result from /api/result/{job_id}
  FAILED      → Show error_message to user

Response 404:
  { "detail": "Job not found" }
```

### GET /api/result/{job_id}

```
Response 200:
  {
    "status": "success",
    "classification": {
       "class_id": 1,
       "label": "AI_VIDEO"
    },
    "flagged_ranges": [
       { "start": 10.0, "end": 28.0, "score": 0.87, "spatial": 0.9, "temporal": 0.85 }
    ],
    "heuristic_score": 0.45,
    "diagnostics": {
       "total_windows_processed": 20,
       "coverage": 0.384
    }
  }

Response 400:
  { "detail": "Result not ready or job failed" }
```

### GET /api/health

```
Response 200:
  {
    "status": "healthy",
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 5060",
    "models_loaded": true
  }
```

## 9.6 FFmpeg Integration — Frame Extraction

```python
# pipeline/preprocessor.py

import subprocess
import os
import cv2
import numpy as np

def stream_crops(video_path: str, window_sec: float = 4.0, stride_sec: float = 2.0, 
                 fps: int = 5, target_size: int = 224):
    """
    Generator that parses frames via FFmpeg natively, isolates faces, and yields 
    batched, scaled crops by memory-efficient streaming window ranges.
    """
    frames_per_window = int(window_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    # Pre-validate file dimensions natively via FFProbe 
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
    dim_out = subprocess.check_output(probe_cmd).decode('utf-8').strip().split(',')
    width, height = int(dim_out[0]), int(dim_out[1])

    # Initiate stream parser
    cmd = [
        'ffmpeg', '-i', video_path, '-f', 'image2pipe',
        '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo',
        '-r', str(fps), '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    buffer = []
    frame_idx = 0
    frame_size = width * height * 3
    
    try:
        while True:
            # Build memory buffer block
            while len(buffer) < frames_per_window:
                raw_data = proc.stdout.read(frame_size)
                if not raw_data:
                    break
                frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                buffer.append(frame)
                
            if len(buffer) < frames_per_window:
                break
                
            window_frames = buffer[:frames_per_window]
            start_sec = frame_idx / fps
            end_sec = start_sec + window_sec
            
            # Simulated crop logic mapping full frames for VRAM constraints:
            stacked_window = []
            for frame in window_frames:
               crop = cv2.resize(frame, (target_size, target_size))
               stacked_window.append(crop)
               
            yield stacked_window, (start_sec, end_sec)
            
            # Slide tracking buffer
            buffer = buffer[stride_frames:]
            frame_idx += stride_frames
            
    finally:
        proc.stdout.close()
        proc.wait()
```

## 9.7 Error Handling

| Error | Where caught | Action |
|---|---|---|
| Invalid file type | API middleware | Return 400, pipeline never runs |
| Unsupported codec | ffprobe check | Return 400, pipeline never runs |
| File too large | API validation | Return 400, pipeline never runs |
| Corrupted video | preprocessor.py | status=FAILED, error_message saved |
| GPU OOM | video_branch.py | Lock queue timeout prevents parallel loads; torch.cuda.empty_cache protects workers |
| Single window fails | analyze_window() | Skip window, mark as NULL, continue |
| No face detected | MTCNN | Use full frame as fallback, face_detected=False, continue |
| Job not found | API routes | Return 404 |
| FFmpeg not installed | preprocessor.py | Raise on startup check, log clear error |

```python
# Startup check for FFmpeg — add to main.py startup:
def check_dependencies():
    deps = ['ffmpeg', 'ffprobe']
    for dep in deps:
        result = subprocess.run(['which', dep], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"'{dep}' not found. Install with: sudo dnf install ffmpeg"
            )
    print("✓ FFmpeg and ffprobe found")

check_dependencies()  # call before app starts
```

---

# 10. Frontend ↔ Backend ↔ AI Connection

## 10.1 Frontend State Machine

```
States:
  IDLE         → Upload form shown, no active job
  UPLOADING    → File being sent to server
  PROCESSING   → Pipeline running (poll for progress)
  COMPLETED    → Result available, render results page
  FAILED       → Show error message, offer retry

Transitions:
  IDLE         → UPLOADING    (user clicks Analyze)
  UPLOADING    → PROCESSING   (server returns job_id)
  PROCESSING   → COMPLETED    (poll returns COMPLETED)
  PROCESSING   → FAILED       (poll returns FAILED or timeout)
  FAILED       → IDLE         (user clicks Try Again)
  COMPLETED    → IDLE         (user clicks Analyze Another)
```

## 10.2 Polling Logic with Error Handling and Backoff

```javascript
// FIX (from v2.0): Full error handling + exponential backoff on network failures
// FIX (from v2.0): Consecutive error counter — 3 failures in a row = abort

function pollStatus(jobId, onProgress, onComplete, onFail) {
  const MAX_WAIT_MS = 600000        // 10 minutes max
  const BASE_INTERVAL_MS = 3000    // poll every 3 seconds
  const startTime = Date.now()
  let currentInterval = BASE_INTERVAL_MS
  let consecutiveErrors = 0
  const MAX_CONSECUTIVE_ERRORS = 3
  let timeoutId = null

  async function poll() {
    if (Date.now() - startTime > MAX_WAIT_MS) {
      onFail("Analysis timed out after 10 minutes. Please try again.")
      return
    }

    try {
      const res = await fetch(`/api/status/${jobId}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      // Reset error state on successful response
      consecutiveErrors = 0
      currentInterval = BASE_INTERVAL_MS
      onProgress(data.progress, data.current_step)

      if (data.status === "COMPLETED") {
        try {
          const resultRes = await fetch(`/api/result/${jobId}`)
          const result = await resultRes.json()
          onComplete(result)
        } catch (e) {
          onFail("Analysis complete but failed to load result. Please refresh.")
        }
        return  // stop polling
      }

      if (data.status === "FAILED") {
        onFail(data.error_message || "Analysis failed. Please try a different video.")
        return  // stop polling
      }

      timeoutId = setTimeout(poll, currentInterval)

    } catch (networkError) {
      consecutiveErrors++

      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        onFail("Connection lost after 3 retries. Check your network and try again.")
        return
      }

      // Exponential backoff: 3s → 6s → 12s
      currentInterval = BASE_INTERVAL_MS * Math.pow(2, consecutiveErrors - 1)
      onProgress(null, `Connection issue — retrying in ${currentInterval / 1000}s...`)
      timeoutId = setTimeout(poll, currentInterval)
    }
  }

  timeoutId = setTimeout(poll, currentInterval)

  // Return cleanup function for component unmount
  return () => clearTimeout(timeoutId)
}
```

## 10.3 UI Pages

### Upload Page

```
┌─────────────────────────────────────────┐
│        DEEPFAKE DETECTION               │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │    Drag & Drop Video Here       │    │
│  │         or click to browse      │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Accepted: MP4, MOV, AVI, MKV          │
│  Max size: 500MB                        │
│  Duration: 30 seconds to 5 minutes      │
│                                         │
│  [ ANALYZE VIDEO ]                      │
└─────────────────────────────────────────┘
```

### Processing Page

```
┌─────────────────────────────────────────┐
│        ANALYZING YOUR VIDEO             │
│                                         │
│  Job ID: abc-123-def-456                │
│                                         │
│  ████████████████░░░░░░░░  65%          │
│                                         │
│  Analyzing window 19 of 29...           │
│                                         │
│  Elapsed: 23 seconds                    │
└─────────────────────────────────────────┘
```

### Results Page — The Demo Moment

```
┌───────────────────────────────────────────────────────┐
│  VERDICT: AI VIDEO DETECTED                           │
│  Confidence: 87%                          [label_index=1]
├───────────────────────────────────────────────────────┤
│  VIDEO TIMELINE (124.5 seconds total)                 │
│                                                       │
│  0s──────────────────────────────────────────────124s │
│  ░░░░░░░░░░▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░▓▓▓▓░░░░░░░░░  │
│                                                       │
│  ▓ = AI detected  ░ = clean                          │
├───────────────────────────────────────────────────────┤
│  FLAGGED MOMENTS                                      │
│                                                       │
│  🎬 0:10 – 0:28   Confidence: 87%  [click to seek]   │
│  🎬 1:12 – 1:21   Confidence: 76%  [click to seek]   │
│                                                       │
│  Face detected in flagged segments: Yes               │
├───────────────────────────────────────────────────────┤
│  VIDEO AUTHENTICITY                                   │
│  █████████░░░░░  61.6% clean  (38.4% flagged)        │
└───────────────────────────────────────────────────────┘

Key feature: Clicking a flagged moment timestamp seeks the video player.
             This is your WOW moment for judges.
```

## 10.4 Timeline Visualization — Implementation Note

```javascript
// Key interaction: timeline bar maps seconds to pixel positions
// Click on timeline → video.currentTime = that second

function TimelineBar({ ranges, duration, videoRef }) {
  const handleTimelineClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const ratio = clickX / rect.width
    const seekTime = ratio * duration
    videoRef.current.currentTime = seekTime
  }

  return (
    <div
      style={{ width: '100%', height: 24, background: '#e5e7eb', cursor: 'pointer', position: 'relative' }}
      onClick={handleTimelineClick}
    >
      {ranges.map((range, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left:   `${(range.start / duration) * 100}%`,
            width:  `${((range.end - range.start) / duration) * 100}%`,
            height: '100%',
            background: `rgba(239, 68, 68, ${0.5 + range.avg_confidence * 0.5})`, // red, opacity by confidence
          }}
          title={`${range.start.toFixed(1)}s – ${range.end.toFixed(1)}s (${(range.avg_confidence * 100).toFixed(0)}% confidence)`}
        />
      ))}
    </div>
  )
}
```

---

# 11. VRAM Management on 8GB GPU

## 11.1 Memory Budget

```
RTX 5060 — 8GB VRAM Total

Allocated at startup (models always resident):
  EfficientNet-B4 weights:     ~250MB
  FrameLSTM weights:         ~430MB
  MTCNN (on CPU):              ~0MB GPU
  ──────────────────────────────────────
  Static overhead:             ~680MB

Per-window inference:
  MTCNN forward pass:          ~100MB (CPU — no GPU cost)
  EfficientNet batch [40,3,224,224]:
    fp32 activations:          ~2.5–3.5GB peak
  After EfficientNet, cache cleared → ~300MB
  FrameLSTM [1,8,3,224,224]:
    fp32 activations:          ~1.5–2.5GB peak
  After FrameLSTM, cache cleared → ~430MB (weights only)
  ──────────────────────────────────────────────────────────────
  Max simultaneous peak:       ~3.5–4.2GB (sequential, not parallel)
  Remaining headroom:          ~3.8–4.5GB (system + other processes)

NEVER run EfficientNet and FrameLSTM in parallel.
Run them sequentially and clear cache between them.
```

## 11.2 Rules for 8GB VRAM

```
RULE 1: Always run models sequentially, never simultaneously.
  Do not: future = executor.submit(run_efficientnet, batch)
          then immediately: run_FrameLSTM(frames_8)
  Do:     run_efficientnet(batch) → del batch → torch.cuda.empty_cache()
          then: run_FrameLSTM(frames_8) → del frames_8 → torch.cuda.empty_cache()

RULE 2: Always use torch.no_grad() during inference.
  Inference without no_grad stores gradient buffers → 2× memory usage.
  with torch.no_grad():
      output = model(input)   ← correct

RULE 3: Delete intermediate tensors explicitly.
  batch = torch.stack(frames).to(device)
  output = model(batch)
  del batch           ← free before next operation
  torch.cuda.empty_cache()

RULE 4: Keep batch sizes within budget.
  EfficientNet-B4 batch size 40 at 224×224 fp32:
    40 × 3 × 224 × 224 × 4 bytes = ~680MB just for input tensor
    Activations through B4 add ~2–3GB more
    Total: ~2.5–3.5GB → fits in 8GB with other overhead
  If OOM: reduce to batch_size=20 (process 40 frames in 2 batches)

RULE 5: Use fp16 (half precision) if fp32 OOMs.
  model = model.half()         ← halves weight memory
  batch = batch.half()         ← halves activation memory
  NOTE: some operations don't support fp16 — wrap in autocast:
    with torch.cuda.amp.autocast():
        output = model(batch)

RULE 6: Monitor VRAM during development.
  import torch
  print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
        f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
```

## 11.3 OOM Recovery Pattern

```python
def safe_model_forward(model, input_tensor, fallback_batch_size=None):
    """
    Runs model forward pass with OOM recovery.
    If OOM: halves batch size and retries once.
    """
    try:
        with torch.no_grad():
            return model(input_tensor)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

        if fallback_batch_size is None:
            # Split batch in half and run sequentially
            n = input_tensor.shape[0]
            half = n // 2
            with torch.no_grad():
                out1 = model(input_tensor[:half])
                out2 = model(input_tensor[half:])
            return torch.cat([out1, out2], dim=0)
        else:
            raise   # let caller handle

# Usage in video_branch.py:
frame_scores = safe_model_forward(efficientnet, batch)
```

## 11.4 Training VRAM Budget

```
During EfficientNet-B4 fine-tuning:
  Forward pass activations:    ~3.5–4.5GB
  Backward pass gradients:     ~3.5–4.5GB (same as forward)
  Model weights + optimizer:   ~500MB
  ─────────────────────────────────────────
  Total during training:       ~7.5–9.5GB

This WILL exceed 8GB at batch_size=32.

Solution — use batch_size=16 for training:
  Forward + backward at batch_size=16:  ~4–5GB
  Weights + optimizer:                  ~500MB
  Total:                                ~4.5–5.5GB → fits

Or use gradient accumulation to simulate larger batch:
  # Simulate batch_size=32 using 2 × batch_size=16
  optimizer.zero_grad()
  for i, (batch, labels) in enumerate(loader):
      loss = criterion(model(batch), labels) / 2   # divide by accumulation steps
      loss.backward()
      if (i + 1) % 2 == 0:
          optimizer.step()
          optimizer.zero_grad()
```

---

# 12. Training Strategy & Checkpointing

## 12.1 Training Order

```
Train in this order (DO NOT try to train all models simultaneously):

1. EfficientNet-B4 (PRIORITY — most important model)
   Start training BEFORE you sleep on Day 2.
   Runs overnight. Fine-tunes in 10–15 hours.

2. FrameLSTM (if EfficientNet results look good)
   Only attempt if EfficientNet training succeeds without OOM.
   If it OOMs → switch to FrameLSTM fallback immediately.
   Do not spend more than 3 hours debugging FrameLSTM OOM.

NEVER train both at once. GPU can't handle both training jobs.
```

## 12.2 EfficientNet-B4 Training Script Skeleton

```python
# train_efficientnet.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
import pandas as pd
from PIL import Image
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpoints/efficientnet'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class WindowDataset(Dataset):
    def __init__(self, csv_path: str, split: str, augment: bool = True):
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frame_dir = row['frame_dir']
        label = float(row['video_fake'])

        # Load all frames in window
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        frames = [Image.open(os.path.join(frame_dir, f)).convert('RGB') for f in frame_files]

        # Apply augmentation during training
        if self.augment:
            frames = [self.augment_frame(f) for f in frames]

        # Convert to tensors [40, 3, 224, 224]
        tensors = [torchvision.transforms.functional.to_tensor(f) for f in frames]
        return torch.stack(tensors), torch.tensor([label])

    def augment_frame(self, img):
        import torchvision.transforms as T
        import random
        transforms = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomHorizontalFlip(p=0.5),
        ])
        return transforms(img)

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)
    print(f"✓ Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✓ Resumed from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    return model, optimizer, start_epoch

def train():
    # Model
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
    head = nn.Linear(model.num_features, 1)
    model.classifier = head   # replace final layer

    # Freeze backbone initially (Phase 1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = WindowDataset('labels/train.csv', 'train', augment=True)
    val_dataset   = WindowDataset('labels/val.csv',   'val',   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False,
                              num_workers=4, pin_memory=True)

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint if exists
    latest_ckpt = f'{CHECKPOINT_DIR}/latest.pth'
    if os.path.exists(latest_ckpt):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, latest_ckpt)

    print(f"Training on {DEVICE}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    for epoch in range(start_epoch, 10):

        # Phase 2 unfreeze at epoch 5
        if epoch == 5:
            print("Unfreezing last 3 blocks for fine-tuning")
            blocks_to_unfreeze = list(model.children())[-4:-1]  # last 3 blocks
            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
            )

        # Training loop
        model.train()
        train_loss = 0.0
        for batch_frames, labels in train_loader:
            # batch_frames: [B, 40, 3, 224, 224] → flatten to [B*40, 3, 224, 224]
            B, N, C, H, W = batch_frames.shape
            frames_flat = batch_frames.view(B * N, C, H, W).to(DEVICE)
            labels = labels.repeat(1, N).view(-1, 1).to(DEVICE)  # repeat label for each frame

            optimizer.zero_grad()
            logits = model(frames_flat)        # [B*N, 1]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            del frames_flat, labels, logits
            torch.cuda.empty_cache()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_frames, labels in val_loader:
                B, N, C, H, W = batch_frames.shape
                frames_flat = batch_frames.view(B * N, C, H, W).to(DEVICE)
                labels = labels.repeat(1, N).view(-1, 1).to(DEVICE)
                logits = model(frames_flat)
                val_loss += criterion(logits, labels).item()
                del frames_flat, labels, logits
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f} val_loss={val_loss:.4f}")

        # Save checkpoint EVERY EPOCH
        save_checkpoint(model, optimizer, epoch, val_loss,
                        f'{CHECKPOINT_DIR}/epoch_{epoch}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss,
                        f'{CHECKPOINT_DIR}/latest.pth')  # overwrite latest

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/efficientnet_b4_finetuned.pth')
            print(f"  ★ New best model saved (val_loss={val_loss:.4f})")

if __name__ == '__main__':
    train()
```

## 12.3 Checkpointing Rules

```
NON-NEGOTIABLE: Save checkpoint after EVERY epoch.

If training crashes at hour 12 without checkpoints → you lose 12 hours.
If training crashes at hour 12 WITH checkpoints → you lose at most 1 epoch.

Checkpoint saves:
  checkpoints/efficientnet/epoch_0.pth
  checkpoints/efficientnet/epoch_1.pth
  ...
  checkpoints/efficientnet/latest.pth  ← always the most recent
  models/efficientnet_b4_finetuned.pth ← best validation loss only

On crash/OOM: script auto-resumes from latest.pth if it exists.
              Always add the resume logic (shown above) before starting.
```

## 12.4 Training Time Estimates on RTX 5060

```
Dataset size: 50 videos per class, batch_size=16

EfficientNet-B4 (batch_size=16, 10 epochs):
  Per epoch with ~5000 windows:  45–75 minutes
  10 epochs total:               7.5–12.5 hours
  Start before sleeping → done by morning ✓

FrameLSTM (if attempted):
  Per epoch:                     60–90 minutes (transformer is slower)
  10 epochs:                     10–15 hours
  VRAM risk: likely OOM during backprop on 8GB
  Strategy: attempt Phase 1 only (frozen backbone, head only)
  If OOM in Phase 1 → switch to FrameLSTM immediately

FrameLSTM (fallback):
  Per epoch:                     15–25 minutes
  10 epochs:                     2.5–4 hours
  No OOM risk whatsoever

Recommendation:
  Start EfficientNet-B4 training on Night 2.
  On Day 3, assess results.
  If accuracy is already acceptable: skip temporal model, ship it.
  If temporal model needed: try FrameLSTM Phase 1 only.
  If FrameLSTM OOMs: FrameLSTM in 2.5–4 hours.
```

## 12.5 Fallback Demo Strategy

```
BEFORE YOU START TRAINING ANYTHING:

1. Download a pretrained FaceForensics++ checkpoint from HuggingFace:
   Model: selimsef/dfdc_deepfake_challenge_solution
   Or:    Wvvijin/FaceForensics (EfficientNet already fine-tuned)

2. Test it on one known deepfake and one real video.

3. Record the result.

WHY: If everything goes wrong on Day 4 — training fails,
     inference pipeline crashes, weights are bad —
     you still have a working demo using pretrained weights.

A WORKING demo with pretrained weights on 5 videos
beats a BROKEN demo with custom weights every time.

Don't skip this step. It costs 30 minutes and could save your hackathon.
```

---

# 13. Fedora 43 Environment Setup

## 13.1 System Dependencies

```bash
# Update system first
sudo dnf upgrade --refresh -y

# Install FFmpeg (required for all video processing)
# Fedora 43 — enable RPM Fusion first if not already done:
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

sudo dnf install ffmpeg ffmpeg-devel -y

# Verify FFmpeg works:
ffmpeg -version
ffprobe -version

# Install Python build tools
sudo dnf install python3-devel python3-pip gcc g++ cmake -y

# Install CUDA toolkit (if not already installed via NVIDIA driver)
# RTX 5060 requires CUDA 12.x or newer
# Check existing CUDA:
nvidia-smi
nvcc --version  # if nvcc not found:
sudo dnf install cuda-toolkit -y
```

## 13.2 Python Environment

```bash
# Create isolated virtual environment
python3 -m venv deepfake_env
source deepfake_env/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA support
# Check CUDA version from nvidia-smi output, then use matching torch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is accessible from PyTorch:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA GeForce RTX 5060
```

## 13.3 Python Dependencies

```txt
# requirements.txt

# Core ML
torch>=2.2.0
torchvision>=0.17.0
torchaudio>=2.2.0
timm>=0.9.12
transformers>=4.38.0
accelerate>=0.27.0

# Face Detection
facenet-pytorch>=2.6.0

# Computer Vision
opencv-python>=4.9.0
Pillow>=10.2.0
numpy>=1.26.0

# AI Video Generation / Injection
insightface>=0.7.3
onnxruntime-gpu>=1.17.0

# Backend
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9
aiofiles>=23.2.1

# Data
pandas>=2.2.0
scikit-learn>=1.4.0

# Utilities
tqdm>=4.66.0
pydantic>=2.6.0
```

```bash
pip install -r requirements.txt
```

## 13.4 Known Fedora 43 Specific Issues

```
ISSUE 1 — facenet-pytorch MTCNN dependency conflict:
  Symptom: ImportError on facenet_pytorch import
  Fix:     pip install facenet-pytorch --no-deps
           pip install requests torch pillow numpy  (install deps manually)

ISSUE 2 — FFmpeg libav not found by Python opencv:
  Symptom: cv2.VideoCapture returns empty frames
  Fix:     sudo dnf install libavcodec-free libavformat-free libswscale-free -y
           Then reinstall opencv: pip uninstall opencv-python && pip install opencv-python

ISSUE 3 — CUDA visible but torch.cuda.is_available() returns False:
  Symptom: nvidia-smi shows GPU but torch can't see it
  Fix:     Check CUDA version mismatch:
           nvidia-smi → shows CUDA version (e.g. 12.4)
           torch install must match: --index-url https://download.pytorch.org/whl/cu124
           Reinstall torch with correct version string.

ISSUE 4 — insightface ONNX Runtime GPU not using CUDA:
  Symptom: InsightFace runs but slowly (using CPU despite GPU)
  Fix:     pip uninstall onnxruntime onnxruntime-gpu
           pip install onnxruntime-gpu
           Verify: python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
           Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']

ISSUE 5 — FrameLSTM HuggingFace download fails / times out:
  Symptom: ConnectionError or timeout during model download
  Fix:     Pre-download before starting the pipeline:
           python -c "from transformers import FrameLSTMModel; FrameLSTMModel.from_pretrained('facebook/FrameLSTM-base-finetuned-k400')"
           Models cache to ~/.cache/huggingface/
           Once cached, no network needed.

ISSUE 6 — uvicorn not finding CUDA during FastAPI startup:
  Symptom: GPU shows available in script but not in uvicorn process
  Fix:     Set CUDA device before starting:
           export CUDA_VISIBLE_DEVICES=0
           uvicorn main:app --host 0.0.0.0 --port 8000
```

## 13.5 Startup Verification Script

```python
# verify_setup.py — run this before starting development

import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

import timm
model = timm.create_model('efficientnet_b4', pretrained=False)
print(f"timm EfficientNet-B4: OK (features={model.num_features})")
del model

from facenet_pytorch import MTCNN
mtcnn = MTCNN()
print("MTCNN: OK")
del mtcnn

from transformers import FrameLSTMModel
print("transformers: OK (FrameLSTM not downloaded yet — run download separately)")

import subprocess
result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
print(f"FFmpeg: {'OK' if result.returncode == 0 else 'NOT FOUND'}")
result2 = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True)
print(f"ffprobe: {'OK' if result2.returncode == 0 else 'NOT FOUND'}")

print("\n✓ All systems verified. Ready to build.")
```

---

# 14. Known Errors, Debugging Guide & Fixes

## 14.1 Shape Errors (Most Common Class of Bug)

```
ERROR: RuntimeError: Expected input batch_size (X) to match target batch_size (Y)
CAUSE: Usually from wrong label broadcasting in training loop
FIX:   Check label shape. If frames are [B*N, 3, 224, 224], labels must be [B*N, 1]
       Not [B, 1]. Use: labels = labels.repeat(1, N).view(-1, 1)

ERROR: RuntimeError: Input size (512) mismatch (768)
CAUSE: Wrong Linear head size for EfficientNet or FrameLSTM
FIX:   Check model.num_features BEFORE defining the head:
       model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
       print(model.num_features)  # should be 1792 for B4
       head = nn.Linear(1792, 1)  # not 512, not 768

ERROR: RuntimeError: Expected 5D tensor for FrameLSTM, got 4D
CAUSE: Missing batch dimension in FrameLSTM input
FIX:   Input must be [1, 8, 3, 224, 224] not [8, 3, 224, 224]
       Add: frames_8_batch = frames_8.unsqueeze(0)

ERROR: RuntimeError: shape mismatch [40, 3, 224, 224] vs [40, 3, 224, 224]
CAUSE: EfficientNet and FrameLSTM both receive 224×224 frames
FIX:   Resize to 224×224 BEFORE passing to FrameLSTM only:
       frames_8_resized = F.interpolate(frames_8, size=(224, 224), mode='bilinear')
```

## 14.2 VRAM / OOM Errors

```
ERROR: torch.cuda.OutOfMemoryError: CUDA out of memory.
       Tried to allocate X GB (GPU Y; Z GiB total capacity)

STEP 1: Find where it crashes:
  Add print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB") at each model call.

STEP 2: Apply fixes in order:
  Fix A: Add torch.cuda.empty_cache() after each model forward call
  Fix B: Reduce batch size from 32 to 16
  Fix C: Add with torch.no_grad(): around all inference code
  Fix D: Use fp16: model = model.half(); input = input.half()
  Fix E: Use gradient checkpointing during training:
         model.gradient_checkpointing_enable()

STEP 3 (training OOM): Use gradient accumulation:
  ACCUMULATION_STEPS = 2
  for i, (batch, labels) in enumerate(loader):
      loss = criterion(model(batch), labels) / ACCUMULATION_STEPS
      loss.backward()
      if (i + 1) % ACCUMULATION_STEPS == 0:
          optimizer.step()
          optimizer.zero_grad()

STEP 4 (FrameLSTM specific OOM): Switch to FrameLSTM (see Section 6.3 fallback)
  Do not spend more than 3 hours debugging FrameLSTM OOM.
```

## 14.3 FFmpeg Errors

```
ERROR: [Errno 2] No such file or directory: 'ffmpeg'
FIX:   sudo dnf install ffmpeg -y (Fedora, requires RPM Fusion)
       Verify: which ffmpeg

ERROR: ffmpeg: Invalid data found when processing input
CAUSE: Corrupted video or unsupported codec variant
FIX:   Pre-validate all videos in Stage 1 with ffprobe.
       Any video ffprobe can't read → reject before Stage 2.

ERROR: Output frames directory empty after extraction
CAUSE: FFmpeg ran but output path didn't exist, or permission issue
FIX:   Always os.makedirs(output_dir, exist_ok=True) before FFmpeg command
       Check FFmpeg stderr: result.stderr (capture_output=True in subprocess)
       Also check: does the output format string match?
         f'{output_dir}/frame_%06d.jpg'  ← correct
         f'{output_dir}/frame_%d.jpg'    ← will produce frame_0.jpg not frame_000000.jpg

ERROR: FFmpeg command hangs (never returns)
CAUSE: FFmpeg waiting for user input (e.g. overwrite confirmation)
FIX:   Always pass -y flag to FFmpeg to auto-confirm overwrites:
       ['ffmpeg', '-i', input, ..., output_path, '-y']
```

## 14.4 Data Pipeline Errors

```
ERROR: KeyError: 'video_id' in injection_log.csv JOIN
CAUSE: Stage 2 didn't run for some videos, or video_id format mismatch
FIX:   Verify injection_log.csv has rows for all Class 1 videos:
       df = pd.read_csv('injection_log.csv')
       print(df[df['class'] == 1].shape)  # should match your Class 1 video count

ERROR: Window frame counts inconsistent (some windows have 38 frames, some 40)
CAUSE: Videos with non-integer-fps or variable frame rate
FIX:   During frame extraction, always specify exact fps:
       -vf fps=10  (not -vf fps=exact or -r 10)
       Then sort frame files by filename before slicing into windows.
       If a window has fewer than expected frames → pad with last frame duplicated.

ERROR: All windows labeled fake=0 for Class 1 (overlap rule not triggering)
CAUSE: injection_log timestamps in seconds, window timestamps in frames (or vice versa)
FIX:   Standardize: ALL timestamps in seconds (floats) throughout the pipeline.
       Never mix frames and seconds in the same comparison.
       Print a few examples: print(f"Window [{win_start},{win_end}] vs inject [{v_inject_start},{v_inject_end}]")

ERROR: data leakage detected (val accuracy too high, ~99%)
CAUSE: Split performed at window level instead of video level
FIX:   Group by video_id before splitting:
       video_ids = df['video_id'].unique()
       train_ids, val_ids = train_test_split(video_ids, test_size=0.2, random_state=42)
       train_df = df[df['video_id'].isin(train_ids)]
       val_df   = df[df['video_id'].isin(val_ids)]
```

## 14.5 Model Training Errors

```
ERROR: Loss becomes NaN after a few batches
CAUSE: Learning rate too high, or gradient explosion
FIX A: Reduce lr from 1e-3 to 1e-4
FIX B: Add gradient clipping:
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
FIX C: Check for NaN in input data:
       assert not torch.isnan(batch).any(), "NaN in input batch"
FIX D: Use BCEWithLogitsLoss not BCELoss (more numerically stable)

ERROR: Validation accuracy stuck at 50% (random)
CAUSE A: Label imbalance — model predicts all-0 or all-1
FIX A:   Add pos_weight to BCEWithLogitsLoss:
         n_neg = (labels == 0).sum().float()
         n_pos = (labels == 1).sum().float()
         criterion = nn.BCEWithLogitsLoss(pos_weight=n_neg/n_pos)

CAUSE B: Learning rate too low — frozen backbone never adapts
FIX B:   Verify unfreezing happened. Print requires_grad for each layer.
         If all False → unfreeze logic is broken.

ERROR: Checkpoint loading fails with "unexpected key" error
CAUSE: Model architecture changed between saves
FIX:   Use strict=False when loading:
       model.load_state_dict(checkpoint['model_state_dict'], strict=False)
       Print missing/unexpected keys to understand the mismatch.

ERROR: Training is very slow (much slower than expected)
CAUSE: DataLoader using 0 workers (single-threaded data loading)
FIX:   Set num_workers=4 in DataLoader (Fedora supports this fine)
       Also set pin_memory=True for faster GPU transfer:
       DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)
```

## 14.6 FastAPI / Backend Errors

```
ERROR: 422 Unprocessable Entity on file upload
CAUSE: Pydantic validation failing on request body
FIX:   File uploads must use File() and UploadFile:
       from fastapi import File, UploadFile
       @app.post("/api/upload")
       async def upload(file: UploadFile = File(...)):

ERROR: Models not loaded when request arrives
CAUSE: Models loaded after app startup, or lifespan not configured
FIX:   Use lifespan context manager (FastAPI 0.107+):
       @asynccontextmanager
       async def lifespan(app):
           load_models()
           yield
       app = FastAPI(lifespan=lifespan)

ERROR: CORS error in browser ("blocked by CORS policy")
CAUSE: FastAPI not configured with CORS middleware
FIX:   from fastapi.middleware.cors import CORSMiddleware
       app.add_middleware(
           CORSMiddleware,
           allow_origins=["http://localhost:3000"],  # React dev server
           allow_methods=["*"],
           allow_headers=["*"],
       )

ERROR: SQLite "database is locked" error under concurrent requests
CAUSE: Multiple requests hitting SQLite simultaneously
FIX A: Use check_same_thread=False in SQLite connection:
       conn = sqlite3.connect('deepfake.db', check_same_thread=False)
FIX B: Use threading.Lock() around all DB writes:
       db_lock = threading.Lock()
       with db_lock:
           conn.execute("UPDATE jobs SET status=? WHERE job_id=?", ...)
FIX C: Since inference is synchronous and blocking anyway,
       concurrent requests will queue naturally. Add a simple semaphore:
       inference_semaphore = asyncio.Semaphore(1)  # 1 at a time
```

## 14.7 InsightFace / Face Detection Errors

```
ERROR: No module named 'insightface'
FIX:   pip install insightface onnxruntime-gpu

ERROR: InsightFace model download fails
FIX:   Models download to ~/.insightface/ on first use.
       If network is slow: download manually and place in that directory.
       Model needed: inswapper_128.onnx (~300MB)

ERROR: MTCNN detects 0 faces in most frames
CAUSE A: Image not RGB (might be BGR from OpenCV)
FIX A:   Convert before MTCNN: img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
CAUSE B: Faces too small in frame
FIX B:   Lower MTCNN threshold: MTCNN(thresholds=[0.5, 0.6, 0.6])
CAUSE C: Video has no visible faces (documentary footage, nature videos)
FIX C:   This is expected — fallback to full frame is correct behavior.
         Log face_detected=False and continue. Do not error.

ERROR: face_detected always True but face crops look wrong (random regions)
CAUSE: MTCNN bounding box coordinates are floats, need int conversion
FIX:   boxes = boxes.astype(int)
       x1, y1, x2, y2 = max(0,boxes[0]), max(0,boxes[1]), boxes[2], boxes[3]
       # Always clamp to image bounds — boxes can be out of bounds
       x2 = min(x2, img.width)
       y2 = min(y2, img.height)
```

---

# 15. Configuration & Environment Variables

```bash
# .env — all configuration in one place

# Storage
TEMP_DIR              = /tmp/deepfake
MODELS_DIR            = ./models
CHECKPOINTS_DIR       = ./checkpoints

# Database
SQLITE_PATH           = ./deepfake.db

# API
HOST                  = 0.0.0.0
PORT                  = 8000
CORS_ORIGINS          = http://localhost:3000,http://localhost:5173

# Upload Limits
MAX_UPLOAD_SIZE_MB    = 500
MAX_VIDEO_DURATION_SEC = 300
MIN_VIDEO_DURATION_SEC = 30
ALLOWED_FILE_TYPES    = mp4,mov,avi,mkv
ALLOWED_CODECS        = h264,hevc,vp8,vp9,av1,mpeg4

# Detection Windowing
WINDOW_SIZE_SEC       = 4.0
STRIDE_SEC            = 2.0
FRAME_SAMPLE_RATE     = 10
FRAME_RESIZE          = 224        # EfficientNet-B4 native resolution
FrameLSTM_FRAMES    = 8          # subsample 40 → 8 (every 5th frame)

# Detection Thresholds
FLAG_THRESHOLD           = 0.60   # min score to flag a window as fake
FULL_COVERAGE_THRESHOLD  = 0.85   # min coverage to classify as FULLY_AI
GAP_TOLERANCE_SEC        = 2.0    # merge flagged windows within this gap
MIN_RANGE_DURATION_SEC   = 1.5    # discard flagged ranges shorter than this
CLASS4_HEURISTIC_THRESHOLD = 0.65 # heuristic score to trigger FULLY_AI_VIDEO

# Model Score Weights
VIDEO_FRAME_WEIGHT    = 0.4       # EfficientNet contribution
VIDEO_TEMPORAL_WEIGHT = 0.6       # FrameLSTM/FrameLSTM contribution

# Model Paths
EFFICIENTNET_MODEL_PATH = ./models/efficientnet_b4_finetuned.pth
FrameLSTM_MODEL_PATH  = ./models/FrameLSTM_finetuned.pth   # optional
FRAMELSTM_MODEL_PATH    = ./models/framelstm_finetuned.pth     # alternative
TEMPORAL_MODEL_TYPE     = FrameLSTM  # or: framelstm

# Hardware
DEVICE                = cuda       # or: cpu
GPU_ID                = 0
```

---

# 16. Limitations & Honest Expectations

## 16.1 Realistic Accuracy Estimates

```
With 50–100 videos per class and fine-tuned EfficientNet-B4:

Class 0 (Real):           85–92%   ← easiest: clean signal, model knows real faces
Class 1 (AI_VIDEO):       78–88%   ← good: FaceForensics++ training transfers well
Class 4 (FULLY_AI_VIDEO): 55–75%   ← hardest: heuristic assists but not fully reliable

Overall system accuracy:  ~73–85%

These numbers assume:
  - 50+ videos per class minimum
  - Diverse face-swap tools in Class 1 training data (not just one tool)
  - Diverse AI video tools in Class 4 (Runway + Pika + SVD, not just one)
  - No heavy post-processing or compression on input videos
```

## 16.2 What The System Cannot Reliably Do

| Limitation | Reason | Note for UI |
|---|---|---|
| Detect Class 4 via model alone | EfficientNet trained on face-swaps, not full generation | Heuristic assists; flag as "experimental" |
| Detect future AI tools | Not in training data | Requires periodic retraining |
| Handle heavily compressed video | Compression destroys subtle artifacts | Show warning if low-quality video detected |
| Detect AI video without human faces | MTCNN falls back to full frame | Lower accuracy on face-free content |
| Work on animated / cartoon content | MTCNN and model not trained on it | Heuristic may misclassify |
| Achieve 100% accuracy on any class | Fundamental ML limitation | Always show confidence score |

## 16.3 The Four Numbers That Matter Most

```
These threshold values control system behavior
more than the AI models themselves:

  FLAG_THRESHOLD           = 0.60  → how sensitive window detection is
  FULL_COVERAGE_THRESHOLD  = 0.85  → what % coverage = "fully AI"
  GAP_TOLERANCE_SEC        = 2.0   → how fragmented reported ranges are
  MIN_RANGE_DURATION_SEC   = 1.5   → minimum reportable fake segment

Tune these on your validation set before demo.
Small changes have large effects on false positive/negative rates.

During the hackathon: If you're seeing too many false positives on real videos,
INCREASE FLAG_THRESHOLD to 0.70.
If you're missing obvious deepfakes, DECREASE to 0.50.
```

## 16.4 What Needs Periodic Maintenance (Post-Hackathon)

```
Every 3–6 months:
  ├── Collect samples from newly released AI video tools
  ├── Fine-tune models on new samples
  ├── Re-evaluate threshold values on updated validation set
  └── Update heuristic calibration constants (Section 8.3)

Monitor in production:
  ├── Average confidence scores across all uploads
  ├── Class distribution (sudden shift → possible new tool not in training)
  └── User feedback and false positive reports
```

---

# 17. Post-Hackathon Production Roadmap

## 17.1 What Gets Added in Phase 2 (Audio Detection)

```
When you're ready to add audio detection, these are the additions:

New Classes:
  2: AI_VOICE         (partial AI audio)
  3: AI_VIDEO_AND_VOICE (both partial)
  5: FULLY_AI_AUDIO   (entire audio is AI)
  6: FULLY_AI_BOTH    (both fully AI)

New Models:
  Wav2Vec 2.0 (facebook/wav2vec2-base fine-tuned on ASVspoof 2021)
  Mel Spectrogram CNN (ResNet-18, lightweight)
  VAD pre-check (webrtcvad — skips Wav2Vec on non-speech)

New Pipeline Step:
  FFmpeg audio extraction at 16kHz → /tmp/audio/{job_id}.wav
  Audio windowing: 4s windows at 16kHz = 64,000 samples (zero-padded)
  Audio branch runs in parallel with video branch (or sequentially on 8GB)

API response:
  "audio" field already in schema — populated with zeros for v3.0
  Simply populate it in Phase 2 — no schema change needed

This is why label indices 0,1,4 were preserved — Phase 2 adds 2,3,5,6
without renumbering anything.
```

## 17.2 Backend Production Migration

```
Hackathon (v3.0):         Production target:
─────────────────         ──────────────────
Sync FastAPI          →   Async FastAPI
No queue              →   Celery workers (x3+)
SQLite                →   PostgreSQL (with indexes)
No cache              →   Redis (job status + result cache)
No rate limiting      →   Atomic Lua rate limiting (per IP)
No auth               →   API key authentication
No reverse proxy      →   NGINX (SSL, rate limit, max body)
Single machine        →   Docker Compose (all services)
No monitoring         →   Flower (Celery dashboard, auth required)
No cleanup task       →   Celery Beat (orphan file cleanup every 30m)

Migration path:
  1. Replace SQLite with PostgreSQL — schema is compatible
  2. Add Redis and Celery — inference logic unchanged, just wrapped in task
  3. Add NGINX config — no application code changes
  4. Add Celery Beat cleanup — copy from Section 8 (v2 reference)
  5. Add API key middleware — one middleware class addition

Each step is independent. You can migrate incrementally.
```

## 17.3 Summary of All Fixes Applied (v1 → v2 → v3)

| # | Issue | Fix Applied | Version |
|---|---|---|---|
| 1 | EfficientNet input was 224×224 | Changed to 224×224 (B4 native resolution) | v2 |
| 2 | FrameLSTM received 40 frames | Subsampled to 8 frames (every 5th) | v2 |
| 3 | No audio track check → worker crash | has_audio flag, audio branch conditionally skipped | v2 |
| 4 | Mel CNN variable T dimension | All audio padded to 64,000 samples → fixed [1,128,126] | v2 |
| 5 | EfficientNet 40 sequential GPU calls | Batched to single [40,3,224,224] forward pass | v2 |
| 6 | No face detection before EfficientNet | MTCNN face crop added; full frame fallback if no face | v2 |
| 7 | overall_confidence undefined | Defined as max(peak_video_score) for fake classes | v2 |
| 8 | Class 4/5 + partial anomaly edge case | secondary_anomalies field added to response schema | v2 |
| 9 | Training window cap mismatch inference | Cap removed entirely from training | v2 |
| 10 | Rate limiting race condition | Atomic Redis Lua script replacing INCR pattern | v2 |
| 11 | Hardcoded estimated_time | Dynamic: queue_length × rolling avg_processing_time | v2 |
| 12 | No codec validation | ffprobe codec check in file_validator.py | v2 |
| 13 | Orphaned temp files on worker crash | Celery Beat 30-minute cleanup task added | v2 |
| 14 | Missing DB indexes | Indexes on status, ip_address, results.job_id | v2 |
| 15 | Flower no authentication | FLOWER_BASIC_AUTH env var + --basic_auth flag | v2 |
| 16 | Wav2Vec2 on non-speech audio → false positives | VAD pre-check, skip if < 20% speech | v2 |
| 17 | coverage vs authenticity naming confusion | Standardized: internal=fraction, API=both percent fields | v2 |
| 18 | Dataset 100GB → not feasible | Reduced to 20GB (~2.85GB per class) | v2 |
| 19 | video_path points to deleted file | Renamed to original_video_path, set NULL after cleanup | v2 |
| 20 | No error handling in frontend polling | try/catch + exponential backoff + consecutive error limit | v2 |
| 21 | 7-class system → audio not buildable in 5 days | Reduced to 3 video-only classes (0, 1, 4) | v3 |
| 22 | Class 4 detection unreliable with faceswap model | Temporal consistency heuristic layer added | v3 |
| 23 | Celery/Redis too complex for hackathon | Sync FastAPI + SQLite as hackathon target | v3 |
| 24 | Training time underestimated | Realistic VRAM-constrained estimates documented | v3 |
| 25 | No fallback strategy for OOM | FrameLSTM fallback for FrameLSTM documented | v3 |
| 26 | No fallback demo strategy | Pretrained checkpoint fallback documented | v3 |
| 27 | No Fedora-specific setup guidance | Full Fedora 43 setup + known issues documented | v3 |
| 28 | Augmentation applied to disk (slow/wasteful) | Augmentation moved to DataLoader __getitem__ | v3 |
| 29 | No debugging catalog | Comprehensive error catalog added (Section 14) | v3 |
| 30 | Dataset 20GB → not feasible in 5 days | Minimum viable: 50 videos per class (~4–6GB) | v3 |

---

*Document Version: 3.0*
*Project: Deepfake & AI-Generated Video Detection Platform*
*Architecture: Video-only, 3-class (0=REAL, 1=AI_VIDEO, 4=FULLY_AI_VIDEO)*
*Stack: Python, FastAPI, SQLite, PyTorch (EfficientNet-B4 + FrameLSTM), FFmpeg, React*
*Target Hardware: NVIDIA RTX 5060 8GB VRAM, Fedora 43 Linux*
*Build Target: 5-day hackathon*
*Audio detection: Deferred to Phase 2 (see Section 17)*
*Changes from v2: 10 new fixes (21–30), audio fully removed, hackathon practicality added throughout*
