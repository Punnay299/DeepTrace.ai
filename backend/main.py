import os
import sys
import threading
import queue
import time
from backend.core.database import update_job_status, mark_job_failed
import threading
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.database import init_db
from backend.pipeline.model_loader import load_all_models
from backend.pipeline.preprocessor import stream_crops
from backend.pipeline.video_branch import score_window
from backend.pipeline.aggregator import merge_flagged_windows
from backend.pipeline.class4_heuristic import compute_class4_heuristic

from backend.api.health import router as health_router
from backend.api.status_and_results import router as status_router
from backend.api.upload import router as upload_router
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database structures atomically
    init_db()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, lstm = load_all_models(device)
    
    # Store directly on app.state to bypass dangerous global dict mutations
    app.state.backbone = backbone
    app.state.lstm = lstm
    app.state.device = device
    
    # Instantiate global execution lock (Strict Single Concurrency)
    app.state.inference_lock = threading.Lock()
    
    # Bounded Job Queue
    app.state.job_queue = queue.Queue(maxsize=5)
    app.state.stop_event = threading.Event()
    
    # Background Worker Thread
    def worker_loop():
        # Setup clean execution context
        while not app.state.stop_event.is_set():
            try:
                job_dict = app.state.job_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            job_id = job_dict.get('job_id')
            temp_path = job_dict.get('temp_path')
            
            # Additional pre-check
            if app.state.stop_event.is_set():
                app.state.job_queue.task_done()
                break
                
            update_job_status(job_id, 'PROCESSING')
            
            try:
                start_time = time.time()
                window_results = []
                heuristic_lumas = []
                
                # We acquire the lock ONLY during PyTorch operations, not before streaming
                for window_frames, (start_sec, end_sec) in stream_crops(temp_path, window_sec=4.0, stride_sec=2.0, fps=5):
                    # Software Watchdog
                    if time.time() - start_time > 180:
                        raise TimeoutError("Inference exceeded maximum 180 seconds limit")
                        
                    with app.state.inference_lock:
                        spatial, temporal = score_window(window_frames, app.state.backbone, app.state.lstm, app.state.device)
                        
                    agg_score = 0.4 * spatial + 0.6 * temporal
                    window_results.append({
                        'start': start_sec, 'end': end_sec, 
                        'score': agg_score, 'spatial': spatial, 'temporal': temporal
                    })
                    heuristic_lumas.append(window_frames)
                    
                if not window_results:
                    raise ValueError("No frames processed from video")
                    
                # Aggregation
                flagged_ranges = merge_flagged_windows(window_results, threshold=0.55, tolerance_sec=2.0)
                heuristic_score = compute_class4_heuristic(heuristic_lumas)
                
                total_duration = window_results[-1]['end']
                flagged_duration = sum([r['end'] - r['start'] for r in flagged_ranges])
                video_coverage = flagged_duration / total_duration if total_duration > 0 else 0
                
                if video_coverage >= 0.85:
                    class_id, label = 4, "FULLY_AI_VIDEO"
                elif len(flagged_ranges) > 0:
                    class_id, label = 1, "AI_VIDEO"
                # Intentional Class 4 override via variance heuristic
                elif heuristic_score >= 0.65:
                    class_id, label = 4, "FULLY_AI_VIDEO"
                else:
                    class_id, label = 0, "REAL"
                    
                final_result = {
                    "status": "success",
                    "classification": {"class_id": class_id, "label": label},
                    "flagged_ranges": flagged_ranges,
                    "heuristic_score": heuristic_score,
                    "diagnostics": {"total_windows_processed": len(window_results), "coverage": video_coverage, "duration": total_duration}
                }
                
                from backend.core.database import save_result_and_complete
                save_result_and_complete(job_id, final_result)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                mark_job_failed(job_id, str(e))
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                app.state.job_queue.task_done()
                
    if not getattr(app.state, 'worker_thread', None):
        app.state.worker_thread = threading.Thread(target=worker_loop, daemon=True)
        app.state.worker_thread.start()
        
    yield
    
    if hasattr(app.state, 'stop_event'):
        app.state.stop_event.set()
        if hasattr(app.state, 'worker_thread'):
            app.state.worker_thread.join(timeout=5.0)
            
    print("--- SHUTDOWN --- Clearing memory...")
    if hasattr(app.state, 'backbone'):
        del app.state.backbone
    if hasattr(app.state, 'lstm'):
        del app.state.lstm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular routes
app.include_router(health_router)
app.include_router(status_router)
app.include_router(upload_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
