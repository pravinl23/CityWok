from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints
from app.core.config import settings
import os

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
    # Ensure directories exist
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to CityWok API"}
