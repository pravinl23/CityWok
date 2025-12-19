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

    # Initialize LMDB databases if in LMDB mode
    USE_LMDB = os.getenv('USE_LMDB', 'false').lower() in ('true', '1', 'yes')

    if USE_LMDB:
        print("\n🔧 Initializing LMDB audio fingerprint matcher...")
        try:
            from app.services.audio_fingerprint_lmdb import audio_matcher_lmdb
            audio_matcher_lmdb.initialize()
            print("✓ LMDB initialization complete")
        except Exception as e:
            print(f"⚠️  LMDB initialization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n🔧 Using pickle mode (legacy)")

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
