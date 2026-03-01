import os
import sqlite3
import json
import uuid

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deepfake.db")

def get_db_connection():
    # check_same_thread=False allows FastAPI threads to share SQLite safely per request context
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    try:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'QUEUED',
                    error_message TEXT
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs(job_id)
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    result_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                )
            ''')
    finally:
        conn.close()

def create_job(job_id: str) -> bool:
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute("INSERT OR IGNORE INTO jobs (job_id, status) VALUES (?, 'QUEUED')", (job_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()

def update_job_status(job_id: str, status: str):
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("UPDATE jobs SET status = ? WHERE job_id = ?", (status, job_id))
    finally:
        conn.close()

def save_result_and_complete(job_id: str, result_dict: dict):
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("INSERT INTO results (result_id, job_id, result_json) VALUES (?, ?, ?)",
                         (str(uuid.uuid4()), job_id, json.dumps(result_dict)))
            conn.execute("UPDATE jobs SET status = 'COMPLETED' WHERE job_id = ?", (job_id,))
    finally:
        conn.close()

def mark_job_failed(job_id: str, error_msg: str):
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("UPDATE jobs SET status = 'FAILED', error_message = ? WHERE job_id = ?", (error_msg, job_id))
    finally:
        conn.close()
