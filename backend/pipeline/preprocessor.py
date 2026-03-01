import subprocess
import numpy as np
import cv2
from facenet_pytorch import MTCNN

def extract_all_frames_ffmpeg(video_path, fps=5):
    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
                 
    try:
        dim_out = subprocess.check_output(probe_cmd).decode('utf-8').strip().split('\n')[0].split(',')
        W_native, H_native = int(dim_out[0].strip()), int(dim_out[1].strip())
    except Exception as e:
        raise RuntimeError(f"FFprobe failed to dissect {video_path}. {e}")
        
    cmd = [
        'ffmpeg', '-i', video_path, '-r', str(fps),
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_video, _ = proc.communicate()
    
    frame_size = W_native * H_native * 3
    num_frames = len(raw_video) // frame_size
    frames = np.frombuffer(raw_video, dtype=np.uint8).reshape((num_frames, H_native, W_native, 3))
    
    return frames

def get_sliding_windows(total_frames, fps=5, window_sec=4.0, stride_sec=2.0):
    windows = []
    frames_per_window = int(window_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    start_idx = 0
    while start_idx + frames_per_window <= total_frames:
        windows.append((start_idx, start_idx + frames_per_window))
        start_idx += stride_frames
        
    return windows

def precompute_crops_to_memory(video_path, window_sec=4.0, stride_sec=2.0, fps=5, target_size=224):
    """
    Extracts whole video into RAM. Detects face tracking per sliding window.
    Returns:
        List of [20, 224, 224, 3] window arrays ready for direct injection into AI.
        List of (start_sec, end_sec) tuples explicitly for aggregation mapping.
    """
    frames_all = extract_all_frames_ffmpeg(video_path, fps=fps)
    total_frames = len(frames_all)
    
    if total_frames == 0:
        return [], []
        
    H, W, _ = frames_all[0].shape
    mtcnn = MTCNN(keep_all=False, device='cpu', select_largest=True) # Run heavily on CPU
    
    indices = get_sliding_windows(total_frames, fps, window_sec, stride_sec)
    
    processed_windows = []
    timestamps = []
    
    for start_idx, end_idx in indices:
        window_frames = frames_all[start_idx:end_idx]
        
        # Simple Tracking: find the strongest face across the 20 frames, take its median bbox
        boxes_collected = []
        for f in window_frames:
            boxes, probs = mtcnn.detect(f)
            if boxes is not None:
                boxes_collected.append(boxes[0])
                
        cropped_frames = []
        
        if len(boxes_collected) > 0:
            median_box = np.median(boxes_collected, axis=0)
            x1, y1, x2, y2 = [int(v) for v in median_box]
            
            # clamp bounds
            x1 = max(0, min(x1, W))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H))
            y2 = max(0, min(y2, H))
            
            # Crop and resize
            for f in window_frames:
                crop = f[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_224 = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                else:
                    crop_224 = cv2.resize(f, (target_size, target_size))
                cropped_frames.append(crop_224)
        else:
            # Full Fallback
            for f in window_frames:
                 crop_224 = cv2.resize(f, (target_size, target_size))
                 cropped_frames.append(crop_224)
                 
        stacked_window = np.stack(cropped_frames)
        processed_windows.append(stacked_window)
        
        start_sec = start_idx / float(fps)
        end_sec = end_idx / float(fps)
        timestamps.append((start_sec, end_sec))
        
    return processed_windows, timestamps
