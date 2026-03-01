import os
import uuid
import sys
import shutil
import torch
import timm

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline.preprocessor import precompute_crops_to_memory
from backend.pipeline.video_branch import score_window
from backend.pipeline.class4_heuristic import compute_class4_heuristic
from backend.pipeline.aggregator import merge_flagged_windows, calculate_verdict

# Global state
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- INIT --- Loading models to {device} in EVAL mode...")
    
    # Backbone
    backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1).to(device)
    # TODO: backbone.load_state_dict(...)
    backbone.eval()
    
    try:
        from scripts.train_framelstm import FrameLSTM
    except ImportError as e:
        sys.exit(f"ImportError: Ensure PYTHONPATH is set. {e}")
        
    # FrameLSTM
    lstm = FrameLSTM(feature_dim=1792).to(device)
    # TODO: lstm.load_state_dict(...)
    lstm.eval()
    
    models['backbone'] = backbone
    models['lstm'] = lstm
    models['device'] = device
    yield
    print("--- SHUTDOWN --- Clearing GPU Cache...")
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/api/upload")
async def process_video(file: UploadFile = File(...)):
    # 1. Save locally 
    temp_id = str(uuid.uuid4())
    os.makedirs('temp', exist_ok=True)
    temp_path = f"temp/{temp_id}.mp4"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Precompute the structural frames in a single CPU sweep
        # Yields a list of sliding windows -> List of np.ndarray(20, 224, 224, 3)
        windows, timestamps = precompute_crops_to_memory(temp_path, window_sec=4.0, stride_sec=2.0, fps=5)
        
        window_results = []
        
        # 3. Synchronous Sliding Evaluation
        for window_frames, (start_sec, end_sec) in zip(windows, timestamps):
            # Passes to the resident GPU branch strictly conforming to 8GB limit
            spatial_score, temporal_score = score_window(
                window_frames, 
                models['backbone'], 
                models['lstm'], 
                models['device']
            )
            
            agg_score = 0.4 * spatial_score + 0.6 * temporal_score
            window_results.append({
                'start': start_sec,
                'end': end_sec,
                'score': agg_score,
                'spatial': spatial_score,
                'temporal': temporal_score
            })
            
        # 4. Aggregation and Heuristics
        flagged_ranges = merge_flagged_windows(window_results, threshold=0.55, tolerance_sec=2.0)
        heuristic_score = compute_class4_heuristic(windows)
        
        # 5. Classification
        class_id, label = calculate_verdict(flagged_ranges, heuristic_score)
        
        return {
            "status": "success",
            "classification": {
                "class_id": class_id,
                "label": label
            },
            "flagged_ranges": flagged_ranges,
            "heuristic_score": heuristic_score,
            "diagnostics": {
                "total_windows_processed": len(windows)
            }
        }
        
    finally:
        # Cleanup physical file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
