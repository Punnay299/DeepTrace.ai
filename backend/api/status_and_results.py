import json
from fastapi import APIRouter, HTTPException
from backend.core.database import get_db_connection

router = APIRouter()

@router.get("/api/status/{job_id}")
async def get_status(job_id: str):
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute("SELECT status, error_message FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            return {"job_id": job_id, "status": row["status"], "error_message": row["error_message"]}
    finally:
        conn.close()

@router.get("/api/result/{job_id}")
async def get_result(job_id: str):
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            if row["status"] != "COMPLETED":
                raise HTTPException(status_code=400, detail="Result not ready or job failed")
                
            cursor = conn.execute("SELECT result_json FROM results WHERE job_id = ?", (job_id,))
            res_row = cursor.fetchone()
            if not res_row:
                raise HTTPException(status_code=404, detail="Result data missing")
                
            return json.loads(res_row["result_json"])
    finally:
        conn.close()
