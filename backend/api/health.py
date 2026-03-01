from fastapi import APIRouter, Request
import torch

router = APIRouter()

@router.get("/api/health")
async def health_check(request: Request):
    # Guard CUDA calls safely for CPU-only systems
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    # Safely probe app state 
    models_loaded = hasattr(request.app.state, 'backbone') and request.app.state.backbone is not None
    device_str = str(getattr(request.app.state, 'device', 'unknown'))
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "models_loaded": models_loaded,
        "device": device_str
    }
