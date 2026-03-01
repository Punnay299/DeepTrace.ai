import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from dataset.dataloader import DeepfakeWindowDataset, collate_fn

import logging

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, freeze_backbone, accum_steps):
    model.train()
    
    # Freeze or unfreeze based on phase
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    else:
        # Phase 2: Unfreeze last blocks
        for param in model.parameters():
            param.requires_grad = True

    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    
    optimizer.zero_grad()
    
    for batch_idx, (batch_tensors, batch_labels, _) in enumerate(progress_bar):
        batch_tensors = batch_tensors.to(device)  # [B, T, C, H, W]
        batch_labels = batch_labels.to(device).unsqueeze(1) # [B, 1]
        
        B, T, C, H, W = batch_tensors.shape
        assert [H, W] == [224, 224], f"Tensors drifted from 224px. Shape: {batch_tensors.shape}"
        x = batch_tensors.view(B*T, C, H, W) # [B*T, C, H, W]
        
        # Mixed Precision Forward
        with torch.amp.autocast('cuda'):
            logits = model(x) # [B*T, 1]
            logits_reshaped = logits.view(B, T, -1).mean(dim=1) # Mean across Time -> [B, 1]
            loss = criterion(logits_reshaped, batch_labels) / accum_steps
            
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accum_steps * B
        
        if not freeze_backbone and batch_idx % 20 == 0:
            print(f" [VRAM Profiling] Peak Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            
        progress_bar.set_postfix({'loss': running_loss / ((batch_idx + 1) * B)})
        
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_tensors, batch_labels, _ in tqdm(dataloader, desc="Validating"):
            batch_tensors = batch_tensors.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            B, T, C, H, W = batch_tensors.shape
            x = batch_tensors.view(B*T, C, H, W)
            
            with torch.amp.autocast('cuda'):
                logits = model(x)
                logits_reshaped = logits.view(B, T, -1).mean(dim=1)
                loss = criterion(logits_reshaped, batch_labels)
                
            running_loss += loss.item() * B
            probs = torch.sigmoid(logits_reshaped)
            preds = (probs > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += B
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--effective_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--freeze_epochs', type=int, default=8, help="Epochs to train only the head before partial unfreeze")
    args = parser.parse_args()
    
    if args.batch_size > 8:
        print("X Fail-fast Error: batch_size > 8 will likely cause OOM on 8GB VRAM. Use a smaller batch_size and configure effective_batch_size.")
        sys.exit(1)
        
    accum_steps = max(1, args.effective_batch_size // args.batch_size)
    print(f"Calculated Accumulation Steps: {accum_steps} (Effective Batch Size: {args.batch_size * accum_steps})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Datasets
    train_ds = DeepfakeWindowDataset(args.train_csv, augment=True, seq_len=20, fps=5, target_size=224)
    val_ds = DeepfakeWindowDataset(args.val_csv, augment=False, seq_len=20, fps=5, target_size=224)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # Model
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # Phase 1: Freeze backbone, Phase 2: Unfreeze
        freeze_backbone = epoch <= args.freeze_epochs
        
        # Adjust learning rate for Phase 2
        if epoch == args.freeze_epochs + 1:
            print("--- Entering Phase 2: Unfreezing Backbone ---")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, scaler, device, epoch, freeze_backbone, accum_steps)
        val_loss, val_acc = validate(model, val_dl, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/efficientnet_b4_best.pth')
            print("Saved Best Model!")
            
        torch.save(model.state_dict(), 'models/efficientnet_b4_latest.pth')

if __name__ == "__main__":
    main()
