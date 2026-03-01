import torch
import torch.nn as nn
import timm
import sys

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available. Cannot run VRAM check.")
        sys.exit(0)

    print("--- CUDA VRAM CI Check (Training Simulation) ---")
    
    # 1. Model init
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1).to(device)
    model.train()
    
    # Simulate unfreezing phase (memory intensive)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()

    # Simulate 1 batch of data: Batch Size = 4, Sequence Length = 20
    # Shape: [4, 20, 3, 224, 224]
    B, T, C, H, W = 4, 20, 3, 224, 224
    
    # Create dummy tensors mapped directly to GPU
    batch_tensors = torch.randn(B, T, C, H, W, device=device)
    batch_labels = torch.randint(0, 2, (B, 1), dtype=torch.float32, device=device)

    print(f"Memory allocated before forward pass: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    try:
        optimizer.zero_grad()
        
        # Flatten for backbone
        x = batch_tensors.view(B * T, C, H, W)
        
        with torch.amp.autocast('cuda'):
            logits = model(x) # [80, 1]
            logits_reshaped = logits.view(B, T, -1).mean(dim=1) # [4, 1]
            loss = criterion(logits_reshaped, batch_labels) / 8 # Simulate accum_steps=8
            
        scaler.scale(loss).backward()
        
        # We simulate the step just for completeness of memory profiling
        scaler.step(optimizer)
        scaler.update()

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak Memory during forward/backward pass: {peak_memory:.2f} GB")
        
        if peak_memory > 7.5:
            print("X FAIL: Peak memory exceeded 7.5GB limit!")
            sys.exit(1)
        else:
            print("✓ PASS: Peak memory is under the 7.5GB limit.")
            
    except torch.cuda.OutOfMemoryError:
        print("X FAIL: OutOfMemoryError encountered!")
        sys.exit(1)
    except Exception as e:
        print(f"X Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
