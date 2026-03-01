import os
import sys
import torch
import timm

def load_all_models(device):
    print(f"--- INIT --- Loading models to {device} in EVAL mode...")
    
    # Load Backbone Once
    backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1).to(device)
    # The models are expected to be at the project root or relative to the working dir
    # For safety, resolve paths relative to this file if it's running reliably,
    # but the existing code used relative 'models/efficientnet_b4_best.pth' which depends on CWD.
    # To be safe, we determine the absolute path to the project root:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    backbone_weights = os.path.join(ROOT_DIR, 'models', 'efficientnet_b4_best.pth')
    
    if os.path.exists(backbone_weights):
        # omitted weights_only=True per user instruction
        backbone.load_state_dict(torch.load(backbone_weights, map_location=device))
    else:
        print(f"Warning: Backbone weights not found at {backbone_weights}")
    backbone.eval()
    
    try:
        from scripts.train_framelstm import FrameLSTM
    except ImportError as e:
        sys.exit(f"ImportError: Ensure PYTHONPATH is set. {e}")
        
    # Load FrameLSTM Once
    lstm = FrameLSTM(feature_dim=1792).to(device)
    lstm_weights = os.path.join(ROOT_DIR, 'models', 'framelstm_best.pth')
    if os.path.exists(lstm_weights):
        # omitted weights_only=True 
        lstm.load_state_dict(torch.load(lstm_weights, map_location=device))
    else:
        print(f"Warning: FrameLSTM weights not found at {lstm_weights}")
    lstm.eval()
    
    return backbone, lstm
