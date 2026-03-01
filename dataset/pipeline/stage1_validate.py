import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = 'dataset/raw'
OUTPUT_CSV = 'dataset/labels/validated_videos.csv'

def get_video_info(filepath):
    """Uses ffprobe to extract video metadata."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(filepath)
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if result.returncode != 0:
            return None, f"ffprobe error: {result.stderr}"
        
        info = json.loads(result.stdout)
        
        video_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
                
        if not video_stream:
            return None, "No video stream found"
            
        format_info = info.get("format", {})
        duration = float(format_info.get("duration", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "")
        
        return {
            "duration": duration,
            "height": height,
            "codec": codec,
            "status": "valid",
            "error": ""
        }, None
        
    except Exception as e:
        return None, str(e)

def main():
    os.makedirs('dataset/labels', exist_ok=True)
    
    classes = {
        'class_0_real': 0,
        'class_1_ai_video': 1,
        'class_4_full_ai_video': 4
    }
    
    records = []
    
    for folder_name, class_idx in classes.items():
        folder_path = Path(RAW_DIR) / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
            
        video_files = list(folder_path.glob("*.mp4")) + list(folder_path.glob("*.mov")) + list(folder_path.glob("*.avi"))
        print(f"Scanning {len(video_files)} files in {folder_name}...")
        
        for v_path in tqdm(video_files):
            # We use filename as video_id for simplicity (strip extension)
            video_id = v_path.stem
            
            info, err = get_video_info(v_path)
            
            if info:
                # Validation rules
                if info['duration'] < 30.0 or info['duration'] > 300.0:
                    info['status'] = 'invalid'
                    info['error'] = f"Duration {info['duration']}s out of bounds (30-300)"
                elif info['height'] < 360:
                    info['status'] = 'invalid'
                    info['error'] = f"Resolution {info['height']}p too low"
                elif info['codec'] not in ['h264', 'h265', 'hevc', 'vp8', 'vp9', 'av1', 'mpeg4']:
                    info['status'] = 'invalid'
                    info['error'] = f"Unsupported codec {info['codec']}"
                    
                records.append({
                    'video_id': video_id,
                    'class': class_idx,
                    'path': str(v_path),
                    'duration': info['duration'],
                    'height': info['height'],
                    'codec': info['codec'],
                    'status': info['status'],
                    'error': info['error']
                })
            else:
                records.append({
                    'video_id': video_id,
                    'class': class_idx,
                    'path': str(v_path),
                    'duration': 0,
                    'height': 0,
                    'codec': "",
                    'status': 'error',
                    'error': err
                })
                
    if not records:
        print("No videos found! Generating an empty CSV schema.")
        df = pd.DataFrame(columns=['video_id', 'class', 'path', 'duration', 'height', 'codec', 'status', 'error'])
    else:
        df = pd.DataFrame(records)
        
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated {OUTPUT_CSV} with {len(df)} records.")
    
    valid_count = len(df[df['status'] == 'valid'])
    print(f"Total valid videos: {valid_count} / {len(df)}")

if __name__ == "__main__":
    main()
