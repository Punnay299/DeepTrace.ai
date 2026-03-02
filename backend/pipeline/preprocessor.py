import subprocess
import numpy as np
import cv2
from facenet_pytorch import MTCNN

def stream_crops(video_path, window_sec=4.0, stride_sec=2.0, fps=5, target_size=224):
    """
    Streams a video and yields sliding windows of fully processed (cropped/resized) frames.
    Never loads the entire video into RAM at once.
    Yields:
        stacked_window: np.ndarray shape (20, 224, 224, 3)
        (start_sec, end_sec): tuple of floats
    """
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
    
    frame_size = W_native * H_native * 3
    frames_per_window = int(window_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    mtcnn = MTCNN(keep_all=False, device='cpu', select_largest=True) # Run heavily on CPU
    
    buffer = []
    frame_idx = 0
    
    try:
        while True:
            # Fill buffer until we have enough for a window
            while len(buffer) < frames_per_window:
                raw_frame = proc.stdout.read(frame_size)
                if not raw_frame or len(raw_frame) < frame_size:
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((H_native, W_native, 3))
                buffer.append(frame)
                
            if len(buffer) < frames_per_window:
                break
                
            window_frames = buffer[:frames_per_window]
            
            # Tracking
            boxes_collected = []
            MAX_DIM = 720
            
            for f in window_frames:
                orig_h, orig_w, _ = f.shape
                scale_ratio = 1.0
                
                if max(orig_h, orig_w) > MAX_DIM:
                    scale_ratio = MAX_DIM / max(orig_h, orig_w)
                    new_w = int(orig_w * scale_ratio)
                    new_h = int(orig_h * scale_ratio)
                    detect_frame = cv2.resize(f, (new_w, new_h))
                else:
                    detect_frame = f
                    
                boxes, probs = mtcnn.detect(detect_frame)
                if boxes is not None:
                    # scale box back to physical dimensions
                    real_box = boxes[0] / scale_ratio
                    boxes_collected.append(real_box)
                    
            cropped_frames = []
            
            if len(boxes_collected) > 0:
                median_box = np.median(boxes_collected, axis=0)
                x1, y1, x2, y2 = [int(v) for v in median_box]
                
                # clamp bounds
                x1 = max(0, min(x1, W_native))
                x2 = max(0, min(x2, W_native))
                y1 = max(0, min(y1, H_native))
                y2 = max(0, min(y2, H_native))
                
                for f in window_frames:
                    crop = f[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_224 = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        crop_224 = cv2.resize(f, (target_size, target_size))
                    cropped_frames.append(crop_224)
            else:
                for f in window_frames:
                     crop_224 = cv2.resize(f, (target_size, target_size))
                     cropped_frames.append(crop_224)
                     
            stacked_window = np.stack(cropped_frames)
            
            start_sec = frame_idx / float(fps)
            end_sec = (frame_idx + frames_per_window) / float(fps)
            
            yield stacked_window, (start_sec, end_sec)
            
            # Step forward
            buffer = buffer[stride_frames:]
            frame_idx += stride_frames
            
    finally:
        proc.stdout.close()
        proc.wait()

