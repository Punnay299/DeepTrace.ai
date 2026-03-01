import torch
import gc
import numpy as np

def compile_tensor(frames_np, device):
    """
    Converts (20, H, W, 3) to (1, 20, 3, 224, 224) 
    Normalizes with ImageNet means exactly as T.Normalize.
    """
    assert frames_np.shape[0] == 20, f"Expected 20 frames, got {frames_np.shape[0]}"
    assert frames_np.shape[1] == 224 and frames_np.shape[2] == 224, f"Expected 224x224, got {frames_np.shape[1]}x{frames_np.shape[2]}"

    # To tensor: [T, H, W, C] -> [T, C, H, W]
    inputs = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
    inputs.div_(255.0)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    inputs.sub_(mean).div_(std)
    
    return inputs


def score_window(frames_np, backbone, lstm, device):
    """
    Evaluates a 20-frame spatial+temporal sequence strictly respecting 8GB VRAM limits.
    Models run resident on GPU; features pass directly to LSTM; VRAM is immediately purged.
    
    Returns: spatial_score (float), temporal_score (float)
    """
    B, T, C, H, W = 1, 20, 3, 224, 224
    
    # Process inputs onto device
    inputs = compile_tensor(frames_np, device)
    inputs = inputs.view(B*T, C, H, W)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            # 1. Spatial Forward Pass
            features = backbone.forward_features(inputs)
            features = backbone.global_pool(features) # [B*T, 1792]
            
            # Predict spatial severity (mean across 20 frames)
            frame_logits = backbone.get_classifier()(features)
            spatial_score = torch.sigmoid(frame_logits).mean().item()
            
            # 2. Reshape and Temporal Forward Pass (Resident on GPU)
            features_seq = features.view(B, T, -1) # [1, 20, 1792]
            temporal_logits = lstm(features_seq)
            temporal_score = torch.sigmoid(temporal_logits).item()
            
    # 3. Explicit Memory Ejection
    # Del inputs and intermediate activations permanently
    del inputs, features, frame_logits, features_seq, temporal_logits
    gc.collect()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return spatial_score, temporal_score
