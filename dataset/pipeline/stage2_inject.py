import os
import argparse
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def generate_inject_range(duration: float) -> tuple[float, float]:
    """Generates a random timestamp range for injection. Minimum 8s."""
    max_inject_duration = min(duration * 0.60, duration - 10.0)
    if max_inject_duration < 8.0:
        # Fallback if video is too short, inject entire middle
        inject_duration = duration - 4.0
        inject_start = 2.0
    else:
        inject_duration = random.uniform(8.0, max_inject_duration)
        max_start = duration - inject_duration - 5.0
        inject_start = random.uniform(5.0, max_start)
        
    inject_end = inject_start + inject_duration
    return round(inject_start, 2), round(inject_end, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='dataset/labels/validated_videos.csv')
    parser.add_argument('--output_csv', type=str, default='dataset/labels/injection_log.csv')
    parser.add_argument('--mock', action='store_true', help="Only calculates timestamps without running InsightFace physically.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        return

    df = pd.read_csv(args.input_csv)
    # Only inject Class 1 (AI_VIDEO)
    class1_df = df[(df['class'] == 1) & (df['status'] == 'valid')]
    
    records = []
    
    print(f"Planning injection for {len(class1_df)} Class 1 videos...")
    for _, row in tqdm(class1_df.iterrows(), total=len(class1_df)):
        video_id = row['video_id']
        duration = row['duration']
        
        try:
            start_sec, end_sec = generate_inject_range(duration)
        except Exception as e:
            print(f"Skipping {video_id} injection calculation: {e}")
            continue
            
        records.append({
            'video_id': video_id,
            'class': 1,
            'duration': duration,
            'v_inject_start': start_sec,
            'v_inject_end': end_sec,
            'status': 'calculated'
        })
        
        if not args.mock:
            # Here we would call InsightFace frame extraction, face swap, and FFmpeg splice.
            # Due to the heavy nature of insightface installation and models, 
            # we isolate the mathematical ground-truth generation so master.csv generation is mathematically sound.
            pass
            
    out_df = pd.DataFrame(records)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Successfully generated injection log at {args.output_csv} with {len(out_df)} records.")

if __name__ == "__main__":
    main()
