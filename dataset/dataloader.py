import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
import cv2
from PIL import Image
import torchvision.transforms as T
import subprocess

class DeepfakeWindowDataset(Dataset):
    """
    On-The-Fly DataLoader mapping `windows_index.csv` to PyTorch tensors.
    Uses decord (default, ±1 frame drift from keyframes) or exact ffmpeg pipe-to-memory
    to read exactly 20 frames at 5 FPS from the physical MP4.
    If face_detected is True, it converts normalized bounding boxes to pixel space,
    crops, and reshapes. Otherwise applies full-frame resize.
    Outputs: [20, 3, target_size, target_size] tensors per index.
    """
    def __init__(self, csv_file, augment=False, seq_len=20, fps=5, target_size=224, seed=42, frame_exact=False):
        self.df = pd.read_csv(csv_file)
        self.augment = augment
        self.seq_len = seq_len
        self.fps = fps
        self.target_size = target_size
        self.frame_exact = frame_exact
        
        # We need consistent seeding for augmentation
        if seed is not None:
             torch.manual_seed(seed)
             np.random.seed(seed)

        # Base transforms applied randomly to all frames in the sequence exactly the same way
        self.spatial_transforms = None
        if augment:
            self.spatial_transforms = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.RandomHorizontalFlip(p=0.5)
            ])
            
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        start_sec = row['start_sec']
        end_sec = row['end_sec']
        label = row['injection_label']
        window_id = row['window_id']
        
        frames_np = None
        duration = self.seq_len / self.fps

        if self.frame_exact:
            # frame_exact: use ffmpeg pipe to memory
            try:
                probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                             "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
                dim_out = subprocess.check_output(probe_cmd).decode('utf-8').strip().split('\n')[0].split(',')
                W_native, H_native = int(dim_out[0].strip()), int(dim_out[1].strip())

                cmd = [
                    'ffmpeg', '-ss', str(start_sec), '-i', video_path,
                    '-t', str(duration), '-r', str(self.fps),
                    '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                raw_video = proc.stdout.read()
                proc.wait()

                frame_size = W_native * H_native * 3
                num_frames = len(raw_video) // frame_size
                frames_np_full = np.frombuffer(raw_video, dtype=np.uint8).reshape((num_frames, H_native, W_native, 3))
                
                # If ffmpeg yielded fewer frames (e.g. end of video), pad. If more, slice.
                if len(frames_np_full) >= self.seq_len:
                    frames_np = frames_np_full[:self.seq_len]
                else:
                    # Pad the rest with the last frame
                    frames_np = np.zeros((self.seq_len, H_native, W_native, 3), dtype=np.uint8)
                    n_valid = len(frames_np_full)
                    if n_valid > 0:
                        frames_np[:n_valid] = frames_np_full
                        frames_np[n_valid:] = frames_np_full[-1]
                    
            except Exception as e:
                raise RuntimeError(f"FFmpeg failed to pipe frames from {video_path}: {e}")
                
            H, W, _ = frames_np[0].shape
        else:
            # default: decord
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
            except Exception as e:
                raise RuntimeError(f"Decord failed to read {video_path}: {e}")

            video_fps = vr.get_avg_fps()
            start_idx = int(start_sec * video_fps)
            interval = video_fps / self.fps
            indices = [int(start_idx + i * interval) for i in range(self.seq_len)]
            indices = [min(idx, len(vr) - 1) for idx in indices]
            
            frames_np = vr.get_batch(indices).asnumpy() # [20, H, W, 3] uint8
            H, W, _ = frames_np[0].shape
        
        # Decide the crop bounds
        if row['face_detected'] == 1 and pd.notna(row['x1']):
            # Un-normalize
            x1 = int(row['x1'] * W)
            y1 = int(row['y1'] * H)
            x2 = int(row['x2'] * W)
            y2 = int(row['y2'] * H)
            
            # Clamp bounds
            x1 = max(0, min(x1, W-1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H-1))
            y2 = max(0, min(y2, H))
            
            # Ensure minimum size
            if x2 - x1 < 16:
                 center_x = (x1 + x2) // 2
                 x1 = max(0, center_x - 8)
                 x2 = min(W, center_x + 8)
            if y2 - y1 < 16:
                 center_y = (y1 + y2) // 2
                 y1 = max(0, center_y - 8)
                 y2 = min(H, center_y + 8)
            
            # Crop frames
            frames_cropped = [f[y1:y2, x1:x2] for f in frames_np]
        else:
            # Full frame fallback
            frames_cropped = frames_np
            
        processed_tensors = []
        seed = np.random.randint(2147483647) # Seed for consistent sequence augmentations if any
        
        for frame in frames_cropped:
            # Resize
            img = cv2.resize(frame, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            pil_img = Image.fromarray(img)
            
            # Apply consistent spatial augmentation
            if self.augment:
                 torch.manual_seed(seed) # Ensure RandomHorizontalFlip triggers identically across frames
                 pil_img = self.spatial_transforms(pil_img)
                 
            # To tensor and normalize
            tensor = self.to_tensor(pil_img)
            processed_tensors.append(tensor)
            
        # Stack into [20, 3, 224, 224]
        tensor_stack = torch.stack(processed_tensors)
        
        return tensor_stack, torch.tensor(label, dtype=torch.float32), window_id

def collate_fn(batch):
    """
    Standard PyTorch collate function.
    Combines a list of tuples (tensor_stack, label, window_id) into batched tensors.
    """
    tensors = []
    labels = []
    window_ids = []
    
    for t, l, w in batch:
        tensors.append(t)
        labels.append(l)
        window_ids.append(w)
        
    return torch.stack(tensors), torch.stack(labels), window_ids
