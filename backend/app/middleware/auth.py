from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
import hashlib
import sqlite3
import os
from datetime import datetime

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

DB_PATH = os.path.join(os.getenv("DATA_DIR", "data"), "api_keys.db")


def init_api_key_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            requests_today INTEGER DEFAULT 0,
            total_requests INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)
    
    conn.commit()
    conn.close()


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def is_valid_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    
    key_hash = hash_api_key(api_key)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT is_active FROM api_keys 
            WHERE key_hash = ? AND is_active = TRUE
        """, (key_hash,))
        
        result = cursor.fetchone()
        
        if result:
            cursor.execute("""
                UPDATE api_keys 
                SET last_used = CURRENT_TIMESTAMP,
                    total_requests = total_requests + 1,
                    requests_today = requests_today + 1
                WHERE key_hash = ?
            """, (key_hash,))
            conn.commit()
        
        conn.close()
        return result is not None
    
    except Exception:
        return False


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> Optional[str]:
    if not api_key:
        return None
    
    if not is_valid_api_key(api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid or inactive API key"
        )
    
    return api_key


def generate_api_key(name: str) -> str:
    import secrets
    
    api_key = f"sk_live_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(api_key)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO api_keys (key_hash, name)
        VALUES (?, ?)
    """, (key_hash, name))
    
    conn.commit()
    conn.close()
    
    return api_key


try:
    init_api_key_db()
except Exception:
    pass

