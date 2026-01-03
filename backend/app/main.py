from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import endpoints
from app.core.config import settings
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.middleware.waf import WAFMiddleware
import os
import traceback

SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[
                FastApiIntegration(),
                StarletteIntegration(),
            ],
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "production"),
        )
    except ImportError:
        pass

app = FastAPI(title="CityWok - Episode Identifier", version="0.1.0")

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(WAFMiddleware)

ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS_STR == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
    max_age=3600,
)
app.include_router(endpoints.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("="*60)
    print("CityWok API - Startup")
    print("="*60)

    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    LAZY_LOAD = os.getenv('LAZY_LOAD_PICKLE', 'false').lower() in ('true', '1', 'yes')
    if LAZY_LOAD:
        print("\n🔧 Using pickle mode with lazy loading (fast startup)")
    else:
        print("\n🔧 Using pickle mode (eager loading)")

    print("="*60)
    print("✓ Startup complete!")
    print("="*60)

@app.get("/")
async def root():
    return {"message": "Welcome to CityWok API"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print(f"Unhandled exception: {exc}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
