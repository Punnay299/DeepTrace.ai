import os
import argparse
import random
import pandas as pd
import shutil
import subprocess
import tempfile
import cv2
import glob
from tqdm import tqdm
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import torch

def generate_inject_range(duration: float) -> tuple[float, float]:
    """Generates a random timestamp range for injection. Minimum 8s."""
    max_inject_duration = min(duration * 0.60, duration - 10.0)
    if max_inject_duration < 8.0:
        inject_duration = duration - 4.0
        inject_start = 2.0
    else:
        inject_duration = random.uniform(8.0, max_inject_duration)
        max_start = duration - inject_duration - 5.0
        inject_start = random.uniform(5.0, max_start)
        
    inject_end = inject_start + inject_duration
    return round(inject_start, 2), round(inject_end, 2)

def get_video_fps(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    if '/' in output:
        num, den = output.split('/')
        return float(num) / float(den)
    return float(output)

def get_source_face(app, face_dir="dataset/synthetic_assets/swap_faces/"):
    assert os.path.exists(face_dir), f"Directory {face_dir} does not exist."
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(face_dir, ext)))
    assert len(files) > 0, f"No source face images found in {face_dir}."
    
    source_path = random.choice(files)
    print(f"Using source face: {source_path}")
    img = cv2.imread(source_path)
    faces = app.get(img)
    assert len(faces) > 0, f"No faces detected in source image {source_path}."
    return faces[0]

def process_video(video_path, inject_start, inject_end, app, swapper, source_face):
    output_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_prev = os.path.join(temp_dir, "chunk_prev.mp4")
        chunk_swap = os.path.join(temp_dir, "chunk_swap.mp4")
        chunk_post = os.path.join(temp_dir, "chunk_post.mp4")
        chunk_swapped = os.path.join(temp_dir, "chunk_swapped.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        final_out = os.path.join(temp_dir, "output.mp4")

        fps = get_video_fps(video_path)
        
        # 1. Split video into chunks
        print(f"Trimming {base_name} chunks...")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-t", str(inject_start),
            "-c", "copy", chunk_prev
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ss", str(inject_start),
            "-to", str(inject_end), "-c:v", "libx264", "-c:a", "aac", chunk_swap
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ss", str(inject_end),
            "-c", "copy", chunk_post
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 2. Extract frames
        print(f"Extracting frames for {base_name}...")
        subprocess.run([
            "ffmpeg", "-y", "-i", chunk_swap,
            os.path.join(frames_dir, "frame_%05d.png")
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 3. Swap faces
        print(f"Swapping faces in {base_name}...")
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        for f in frame_files:
            img = cv2.imread(f)
            faces = app.get(img)
            res = img.copy()
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                res = swapper.get(res, faces[0], source_face, paste_back=True)
            cv2.imwrite(f, res)
            
        # 4. Reconstruct video segment
        print(f"Reassembling injected segment for {base_name}...")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), "-i",
            os.path.join(frames_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", chunk_swapped
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        chunk_swapped_audio = os.path.join(temp_dir, "chunk_swapped_audio.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", chunk_swapped, "-i", chunk_swap,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", chunk_swapped_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 5. Concat all chunks
        print(f"Concatenating chunks for {base_name}...")
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, 'w') as f:
            f.write(f"file '{chunk_prev}'\n")
            f.write(f"file '{chunk_swapped_audio}'\n")
            f.write(f"file '{chunk_post}'\n")
            
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "copy", "-c:a", "copy", final_out
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        persistent_out = os.path.join(output_dir, f"{base_name}_temp_final.mp4")
        shutil.copy2(final_out, persistent_out)
        
        return persistent_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='dataset/labels/validated_videos.csv')
    parser.add_argument('--output_csv', type=str, default='dataset/labels/injection_log.csv')
    parser.add_argument('--mock', action='store_true', help="Only calculates timestamps without running InsightFace physically.")
    parser.add_argument('--video', type=str, default=None, help="Process a single specific video_id")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        return

    df = pd.read_csv(args.input_csv)
    # Only inject Class 1 (AI_VIDEO)
    class1_df = df[(df['class'] == 1) & (df['status'] == 'valid')]
    
    if args.video:
        class1_df = class1_df[class1_df['video_id'] == args.video]
        if class1_df.empty:
            print(f"Video {args.video} not found in valid Class 1 records.")
            return
            
    # Always load existing log to append/update the boolean flag cleanly
    if os.path.exists(args.output_csv):
        log_df = pd.read_csv(args.output_csv)
    else:
        log_df = pd.DataFrame(columns=['video_id', 'class', 'duration', 'v_inject_start', 'v_inject_end', 'status'])
        
    if 'swapped' not in log_df.columns:
        log_df['swapped'] = False
        
    app = None
    swapper = None
    source_face = None
    
    if not args.mock:
        print("Initializing InsightFace analyzers...")
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
        assert os.path.exists(model_path), f"CRITICAL: {model_path} does not exist. Ensure inswapper is manually placed."
        swapper = insightface.model_zoo.get_model(model_path, providers=["CUDAExecutionProvider"], download=False)
        assert swapper is not None, "Failed to load inswapper model."
        
        source_face = get_source_face(app)
        
    print(f"Planning injection for {len(class1_df)} videos...")
    for _, row in tqdm(class1_df.iterrows(), total=len(class1_df)):
        video_id = row['video_id']
        duration = row['duration']
        video_path = row['path']
        
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found on disk, skipping.")
            continue
            
        # Check if we already have this video logged
        existing_idx = log_df.index[log_df['video_id'] == video_id].tolist()
        if existing_idx:
            idx = existing_idx[0]
            start_sec = log_df.at[idx, 'v_inject_start']
            end_sec = log_df.at[idx, 'v_inject_end']
            if log_df.at[idx, 'swapped']:
                print(f"Skipping {video_id}: already marked swapped=True.")
                continue
        else:
            try:
                start_sec, end_sec = generate_inject_range(duration)
            except Exception as e:
                print(f"Skipping {video_id} range calculation: {e}")
                continue
                
            new_row = {
                'video_id': video_id,
                'class': 1,
                'duration': duration,
                'v_inject_start': start_sec,
                'v_inject_end': end_sec,
                'status': 'calculated',
                'swapped': False
            }
            log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
            idx = len(log_df) - 1

        if args.mock:
            print(f"[MOCK] Would swap {video_id} between {start_sec}s and {end_sec}s")
            continue
            
        backup_path = f"{video_path}.bak"
        assert not os.path.exists(backup_path), f"Backup {backup_path} already exists! Aborting {video_id} to prevent data loss."
        
        try:
            persistent_out = process_video(video_path, start_sec, end_sec, app, swapper, source_face)
            
            print(f"Backing up {video_path} -> {backup_path}")
            shutil.move(video_path, backup_path)
            
            print(f"Deploying final injection -> {video_path}")
            shutil.move(persistent_out, video_path)
            
            log_df.at[idx, 'swapped'] = True
            log_df.at[idx, 'status'] = 'completed'
            
        except Exception as e:
            print(f"FAILED injection for {video_id}: {e}")
            log_df.at[idx, 'status'] = f"failed: {e}"
        finally:
            # Memory cleanup per video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # Write to log incrementally
        log_df.to_csv(args.output_csv, index=False)
            
    print(f"\nProcessing complete. Log updated at {args.output_csv}.")

if __name__ == "__main__":
    main()
