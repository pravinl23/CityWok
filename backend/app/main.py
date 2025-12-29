from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import endpoints
from app.core.config import settings
import os
import traceback

app = FastAPI(title="CityWok - Episode Identifier", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(endpoints.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("="*60)
    print("CityWok API - Startup")
    print("="*60)

    # Ensure directories exist
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Initialize pickle-based audio fingerprint matcher
    LAZY_LOAD = os.getenv('LAZY_LOAD_PICKLE', 'false').lower() in ('true', '1', 'yes')
    if LAZY_LOAD:
        print("\n🔧 Using pickle mode with lazy loading (fast startup)")
    else:
        print("\n🔧 Using pickle mode (eager loading)")
    # Databases will be loaded in AudioFingerprinter.__init__ (eager or lazy based on env var)

    print("="*60)
    print("✓ Startup complete!")
    print("="*60)

@app.get("/")
async def root():
    return {"message": "Welcome to CityWok API"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch and log all errors."""
    import traceback
    print(f"Unhandled exception: {exc}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
