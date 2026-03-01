import os
import csv
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from decord import VideoReader, cpu

# Hide facenet-pytorch warnings
import warnings
warnings.filterwarnings("ignore")

from facenet_pytorch import MTCNN

# Setup logging
logging.basicConfig(
    filename='dataset/pipeline/pipeline_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WINDOW_SEC = 4.0
STRIDE_SEC = 2.0
FPS = 5

def check_overlap(win_start, win_end, inject_start, inject_end):
    """Returns True if window overlaps with an injection."""
    if pd.isna(inject_start) or pd.isna(inject_end):
        return False
    return (win_end > inject_start) and (win_start < inject_end)

def process_video(video_path, video_id, class_label, injection_df, mtcnn):
    """
    Processes a single video into 4s windows.
    Extracts frames via Decord, runs MTCNN, and calculates median normalized bounding box.
    Returns list of dictionaries containing CSV row data.
    """
    rows = []
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        logger.error(f"Cannot read {video_path}: {e}")
        return rows
        
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration_sec = total_frames / video_fps
    
    # Calculate target indices for 5 FPS reading
    frame_interval = video_fps / FPS 
    
    # Get injection timestamps if this is Class 1
    inject_start, inject_end = None, None
    if class_label == 1 and injection_df is not None:
        inj_row = injection_df[injection_df['video_id'] == video_id]
        if not inj_row.empty:
            inject_start = inj_row.iloc[0]['v_inject_start']
            inject_end = inj_row.iloc[0]['v_inject_end']
    
    # Iterate through windows
    win_start_sec = 0.0
    window_id = 0
    
    while win_start_sec + WINDOW_SEC <= duration_sec:
        win_end_sec = win_start_sec + WINDOW_SEC
        
        # Calculate frame indices for this window at 5 FPS
        start_frame_idx = int(win_start_sec * video_fps)
        indices = [int(start_frame_idx + i * frame_interval) for i in range(int(WINDOW_SEC * FPS))]
        indices = [min(idx, total_frames - 1) for idx in indices] # Clamp indices
        
        try:
            frames = vr.get_batch(indices).asnumpy()
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_id} at {win_start_sec}s: {e}")
            win_start_sec += STRIDE_SEC
            continue
            
        height, width, _ = frames[0].shape
        
        # MTCNN processing
        window_boxes = []
        window_confs = []
        
        for frame in frames:
            boxes, probs = mtcnn.detect(frame)
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob is not None and prob > 0.0:
                        window_boxes.append(box)
                        window_confs.append(prob)
                        
        face_detected = 0
        x1_norm, y1_norm, x2_norm, y2_norm = "", "", "", ""
        conf_mean = ""
        notes = ""

        if window_boxes:
            # We have multiple detections. Calculate promptinence score for each track.
            # Simplified approach: Since we can't easily track IDs across the 5fps frames without a tracker,
            # we group boxes by spatial proximity (IoU or center distance) if needed.
            # However, for a single person in frame mostly, taking the median of all high-conf boxes works well.
            # Let's use a simpler heuristic: just take the median of all valid boxes in the window.
            # This handles slight movement smoothly.
            
            valid_boxes = np.array(window_boxes)
            valid_confs = np.array(window_confs)
            
            # Median smoothing
            median_box = np.median(valid_boxes, axis=0)
            
            x1, y1, x2, y2 = median_box
            
            # Normalize and clamp to [0.0, 1.0]
            x1_n = np.clip(x1 / width, 0.0, 1.0)
            y1_n = np.clip(y1 / height, 0.0, 1.0)
            x2_n = np.clip(x2 / width, 0.0, 1.0)
            y2_n = np.clip(y2 / height, 0.0, 1.0)
            
            # Validity check
            if x2_n > x1_n and y2_n > y1_n:
                face_detected = 1
                x1_norm = round(x1_n, 4)
                y1_norm = round(y1_n, 4)
                x2_norm = round(x2_n, 4)
                y2_norm = round(y2_n, 4)
                conf_mean = round(float(np.mean(valid_confs)), 4)
            else:
                notes = "Invalid clamped bbox dimensions"

        # Cross-check labels
        injection_label = 0
        if class_label == 1:
            injection_label = 1 if check_overlap(win_start_sec, win_end_sec, inject_start, inject_end) else 0
        elif class_label == 4:
            injection_label = 1 # Fully AI
            
        rows.append({
            'video_id': video_id,
            'window_id': f"{video_id}_w{window_id}",
            'start_sec': round(win_start_sec, 2),
            'end_sec': round(win_end_sec, 2),
            'face_detected': face_detected,
            'x1': x1_norm,
            'y1': y1_norm,
            'x2': x2_norm,
            'y2': y2_norm,
            'conf_mean': conf_mean,
            'frame_count_checked': len(frames),
            'injection_label': injection_label,
            'video_path': video_path,
            'notes': notes
        })
        
        win_start_sec += STRIDE_SEC
        window_id += 1
        
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to validated_videos.csv')
    parser.add_argument('--injection_log', type=str, default=None, help='Path to injection_log.csv for Class 1 tracking')
    parser.add_argument('--output_csv', type=str, default='dataset/labels/windows_index.csv')
    parser.add_argument('--device', type=str, default='cuda', help='Device for MTCNN: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load inputs
    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        return

    val_df = pd.read_csv(args.input_csv)
    
    inj_df = None
    if args.injection_log and os.path.exists(args.injection_log):
         inj_df = pd.read_csv(args.injection_log)

    # Initialize MTCNN
    print(f"Initializing MTCNN on {args.device}...")
    mtcnn = MTCNN(keep_all=True, device=args.device, select_largest=True)

    all_windows = []
    
    print(f"Processing {len(val_df)} videos...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        video_path = row['path']
        video_id = row['video_id']
        class_label = row['class']
        
        if not os.path.exists(video_path):
            logger.error(f"Video file missing: {video_path}")
            continue
            
        windows = process_video(video_path, video_id, class_label, inj_df, mtcnn)
        all_windows.extend(windows)

    # Save output
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df = pd.DataFrame(all_windows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Successfully generated {len(out_df)} windows to {args.output_csv}")

if __name__ == "__main__":
    main()
