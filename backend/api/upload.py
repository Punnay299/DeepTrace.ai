import os
import time
import uuid
import shutil
import subprocess
import json
import torch
from fastapi import APIRouter, File, UploadFile, Request, HTTPException

from backend.core.database import create_job, update_job_status, save_result_and_complete, mark_job_failed
from backend.pipeline.preprocessor import stream_crops
from backend.pipeline.video_branch import score_window
from backend.pipeline.class4_heuristic import compute_class4_heuristic
from backend.pipeline.aggregator import merge_flagged_windows

router = APIRouter()

def validate_video(file_path: str):
    # Enforce file size limit immediately
    if os.path.getsize(file_path) > 500 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 500MB limit.")
        
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", file_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    # Check return code before parsing JSON
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail="Invalid video file format or corrupt file.")
        
    info = json.loads(result.stdout)
    video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    
    # Must explicitly contain a video stream
    if not video_stream:
        raise HTTPException(status_code=400, detail="No video stream found in file.")
        
    # Enforce strict resolution bounds (Max 1920x1080 or 1080x1920)
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if max(width, height) > 1920 or min(width, height) > 1080:
        raise HTTPException(status_code=400, detail="Video resolution exceeds 1920x1080 (or 1080x1920) limit.")

    # Safe duration parsing
    try:
        dur_str = video_stream.get("duration", info.get("format", {}).get("duration", "N/A"))
        if dur_str == "N/A":
            raise ValueError("Duration is N/A")
        duration = float(dur_str)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Could not safely parse video duration.")
        
    if duration > 60.0:
        raise HTTPException(status_code=400, detail="Video duration exceeds 60s limit.")

    # FPS Safety Guard
    fps_str = video_stream.get("r_frame_rate", "0/0")
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0
        else:
            fps = float(fps_str)
    except (ValueError, TypeError):
        fps = 0
    if fps > 60.0:
        raise HTTPException(status_code=400, detail="Video exceeds 60 FPS limit.")

@router.post("/api/upload")
async def process_video(request: Request, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    os.makedirs('temp', exist_ok=True)
    temp_path = f"temp/{job_id}.mp4"
    
    # 500MB hard stream limit
    MAX_FILE_SIZE = 500 * 1024 * 1024
    cumulative_bytes = 0
    
    try:
        if not create_job(job_id):
            raise HTTPException(status_code=409, detail="Job ID collision occurred.")
            
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024): # 1MB chunks
                cumulative_bytes += len(chunk)
                if cumulative_bytes > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File too large. Max 500MB limit.")
                buffer.write(chunk)
                
        validate_video(temp_path)
        
        job_dict = {
            "job_id": job_id,
            "temp_path": temp_path
        }
        
        import queue
        request.app.state.job_queue.put_nowait(job_dict)
        
        return {"job_id": job_id, "status": "QUEUED"}
        
    except queue.Full:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=429, detail="The processing queue is currently full. Please try again later.")
    except HTTPException:
        mark_job_failed(job_id, "Validation failed")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        mark_job_failed(job_id, str(e))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail="Internal processing error")
