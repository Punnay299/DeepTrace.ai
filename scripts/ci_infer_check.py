import torch
import torch.nn as nn
import timm
import sys
import time

class FrameLSTM(nn.Module):
    def __init__(self, feature_dim=1792, hidden_dim=256, num_layers=2):
        super(FrameLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_timestep_out = lstm_out[:, -1, :] 
        out = self.fc(last_timestep_out) 
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available. Cannot run Inference Check.")
        sys.exit(0)

    print("--- CUDA Infer CI Check ---")
    
    # 1. Mount Models sequentially to GPU (in eval mode)
    efficientnet = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1).to(device)
    efficientnet.eval()
    
    framelstm = FrameLSTM(feature_dim=1792).to(device)
    framelstm.eval()

    # Base footprint
    print(f"Memory allocated after model init: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    num_windows = 5
    latencies = []

    # Simulate 5 consecutive sliding windows to check memory climbing
    try:
        # Dummy loop
        for i in range(num_windows):
            start_time = time.time()
            # 20 frames at 224x224 (representing MTCNN extracted face crops)
            B, T, C, H, W = 1, 20, 3, 224, 224
            
            # Send window directly to GPU
            window_frames = torch.randn(B, T, C, H, W, device=device)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    x = window_frames.view(B * T, C, H, W)
                    
                    # 1. Feature extraction
                    features = efficientnet.forward_features(x)
                    features = efficientnet.global_pool(features) # [B*T, 1792]
                    features = features.view(B, T, -1)            # [B, T, 1792]
                    
                    # 2. LSTM evaluation directly on GPU
                    temporal_logit = framelstm(features)
                    temporal_score = torch.sigmoid(temporal_logit).item() # scalar to CPU
                    
                    # Also compute the spatial score here if needed by simulating the classifier head
                    frame_logits = efficientnet.get_classifier()(features.view(B*T, -1))
                    frame_score = torch.sigmoid(frame_logits).mean().item()
                    
            # 3. Explicit Memory Cleanup
            del window_frames, x, features, temporal_logit, frame_logits
            torch.cuda.empty_cache()
            
            latencies.append((time.time() - start_time) * 1000)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"Peak Memory during Inference: {peak_memory:.2f} GB")
        print(f"Average latency per 4s window (20 frames): {avg_latency:.2f} ms")

        # Thresholds
        if peak_memory > 6.5:
            print("X FAIL: Inference peak memory exceeded 6.5GB limit!")
            sys.exit(1)
        
        # Real life with MTCNN and decoding will add more, 
        # so keeping pure inference under 150ms is ideal
        if avg_latency > 750:
            print("X FAIL: Latency exceeds 750ms per window!")
            sys.exit(1)
            
        print("✓ PASS: Inference footprint verified safe.")

    except torch.cuda.OutOfMemoryError:
        print("X FAIL: OutOfMemoryError encountered!")
        sys.exit(1)
    except Exception as e:
        print(f"X Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
