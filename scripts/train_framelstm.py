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

class FrameLSTM(nn.Module):
    def __init__(self, feature_dim=1792, hidden_dim=256, num_layers=2):
        super(FrameLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [B, T, feature_dim]
        lstm_out, _ = self.lstm(x) # lstm_out: [B, T, hidden_dim]
        # We take the output of the last timestep
        last_timestep_out = lstm_out[:, -1, :] # [B, hidden_dim]
        out = self.fc(last_timestep_out) # [B, 1]
        return out

def train_lstm(model, backbone, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    
    # We do NOT accumulate gradients here since LSTM is much lighter and fits easily.
    for batch_idx, (batch_tensors, batch_labels, _) in enumerate(progress_bar):
        batch_tensors = batch_tensors.to(device)  # [B, T, C, H, W]
        batch_labels = batch_labels.to(device).unsqueeze(1) # [B, 1]
        
        B, T, C, H, W = batch_tensors.shape
        assert [H, W] == [224, 224], f"Tensors drifted from 224px. Shape: {batch_tensors.shape}"
        x = batch_tensors.view(B*T, C, H, W)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            # Extract features from frozen backbone using mixed precision
            with torch.amp.autocast('cuda'):
                features = backbone.forward_features(x) # [B*T, num_features, H', W']
                features = backbone.global_pool(features) # [B*T, num_features]
                if backbone.drop_rate > 0.:
                     features = torch.nn.functional.dropout(features, p=backbone.drop_rate, training=False)
                
                # Reshape back to sequence
                features = features.view(B, T, -1) # [B, T, 1792]
                
        # Free input frames early on the GPU before passing through the LSTM
        del batch_tensors, x
        
        # Train LSTM
        with torch.amp.autocast('cuda'):
            logits = model(features) # [B, 1]
            loss = criterion(logits, batch_labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * B
        progress_bar.set_postfix({'loss': running_loss / ((batch_idx + 1) * B)})
        
    return running_loss / len(dataloader.dataset)

def validate_lstm(model, backbone, dataloader, criterion, device):
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
                features = backbone.forward_features(x)
                features = backbone.global_pool(features)
                features = features.view(B, T, -1)
                
                logits = model(features)
                loss = criterion(logits, batch_labels)
                
            running_loss += loss.item() * B
            probs = torch.sigmoid(logits)
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
    parser.add_argument('--backbone_weights', type=str, required=True, help="Path to tuned EfficientNet-B4")
    parser.add_argument('--batch_size', type=int, default=16, help="LSTM fits bigger batch")
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Backbone (Frozen)
    print("Loading Frozen EfficientNet Backbone...")
    backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
    
    try:
        backbone.load_state_dict(torch.load(args.backbone_weights, map_location=device))
        print("Successfully loaded tuned backbone weights.")
    except Exception as e:
        print(f"Warning: Failed to load backbone weights {args.backbone_weights}. Proceeding with random mapping. Error: {e}")
        
    backbone = backbone.to(device)
    backbone.eval() # Freeze entirely for feature extraction
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Define Model
    print("Initializing FrameLSTM...")
    model = FrameLSTM(feature_dim=1792, hidden_dim=256, num_layers=2)
    model = model.to(device)
    
    # Datasets
    train_ds = DeepfakeWindowDataset(args.train_csv, augment=True, seq_len=20, fps=5, target_size=224)
    val_ds = DeepfakeWindowDataset(args.val_csv, augment=False, seq_len=20, fps=5, target_size=224)
    
    # Note: Using larger batch size since LSTM training consumes much less VRAM (features are computed no_grad)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    os.makedirs('models', exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_lstm(model, backbone, train_dl, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate_lstm(model, backbone, val_dl, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/framelstm_best.pth')
            print("Saved Best LSTM Model!")
            
        torch.save(model.state_dict(), 'models/framelstm_latest.pth')

if __name__ == "__main__":
    main()
