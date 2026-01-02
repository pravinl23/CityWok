from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import re
from typing import List, Pattern

BLOCKED_PATTERNS: List[Pattern] = [
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
    re.compile(r"\.\.\/", re.IGNORECASE),
    re.compile(r"\.\.\\", re.IGNORECASE),
    re.compile(r"union.*select", re.IGNORECASE),
    re.compile(r"exec\s*\(", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
    re.compile(r"system\s*\(", re.IGNORECASE),
    re.compile(r"passthru\s*\(", re.IGNORECASE),
    re.compile(r"shell_exec", re.IGNORECASE),
    re.compile(r"base64_decode", re.IGNORECASE),
]

BLOCKED_USER_AGENTS: List[Pattern] = [
    re.compile(r"sqlmap", re.IGNORECASE),
    re.compile(r"nikto", re.IGNORECASE),
    re.compile(r"nmap", re.IGNORECASE),
    re.compile(r"masscan", re.IGNORECASE),
    re.compile(r"acunetix", re.IGNORECASE),
]


class WAFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        url = str(request.url)
        
        for pattern in BLOCKED_PATTERNS:
            if pattern.search(url):
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: Malicious pattern detected"
                )
        
        user_agent = request.headers.get("user-agent", "")
        for pattern in BLOCKED_USER_AGENTS:
            if pattern.search(user_agent):
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: Blocked user agent"
                )
        
        if len(url) > 2000:
            raise HTTPException(
                status_code=414,
                detail="URI Too Long"
            )
        
        if len(request.query_params) > 50:
            raise HTTPException(
                status_code=400,
                detail="Too many query parameters"
            )
        
        response = await call_next(request)
        return response

