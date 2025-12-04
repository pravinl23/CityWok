# CityWok 🍜

**Identify South Park episodes from video clips using AI.**

Upload a TikTok, YouTube clip, or any video file and CityWok will tell you which episode it's from and the exact timestamp.

## Features

- 🎬 **Visual Analysis**: CLIP embeddings with keyframe extraction (50-150 frames vs 1800/min)
- 🎵 **Audio Fingerprinting**: Shazam-style spectral peak landmarks
- 🔀 **Hybrid Confirmation**: Cross-verifies both modalities for high accuracy
- 🚀 **GPU Accelerated**: CUDA (AWS) and MPS (Mac) support
- 🐳 **Docker Ready**: Deploy to AWS/GCP with one command

## Quick Start (Local)

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Ingest Episodes

```bash
cd backend
source venv/bin/activate

# Single season
python ingest_episodes.py "/path/to/Season 1" 1

# Multiple seasons
python ingest_episodes.py "/path/to/Season 1" 1 "/path/to/Season 2" 2

# Audio-only (if video already ingested)
python ingest_episodes.py "/path/to/Season 1" 1 --audio-only
```

## Deploy to AWS

### Option 1: Docker (Recommended)

```bash
# CPU deployment
docker-compose up api

# GPU deployment (requires nvidia-docker)
docker-compose up api-gpu
```

### Option 2: EC2 / ECS

1. Build and push to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker build -t citywok-api backend/
docker tag citywok-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/citywok-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/citywok-api:latest
```

2. Deploy to ECS or run on EC2:
```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/citywok-api:latest
```

### GPU Instance Recommendations

| Use Case | Instance | Cost |
|----------|----------|------|
| Dev/Test | g4dn.xlarge | ~$0.50/hr |
| Production | g5.xlarge | ~$1.00/hr |
| High Traffic | g5.2xlarge | ~$2.00/hr |

## API Endpoints

```
POST /api/v1/identify    - Identify episode from video/audio file or URL
POST /api/v1/ingest      - Add episode to database
GET  /api/v1/stats       - Database statistics
GET  /api/v1/test        - Health check
```

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────┐
│   Client    │────▶│              FastAPI                    │
└─────────────┘     │  ┌─────────────────────────────────────┐│
                    │  │      /identify Endpoint             ││
                    │  │  ┌─────────┐    ┌────────────────┐  ││
                    │  │  │ Visual  │    │     Audio      │  ││
                    │  │  │ (CLIP)  │    │ (Fingerprint)  │  ││
                    │  │  └────┬────┘    └───────┬────────┘  ││
                    │  │       └──────┬──────────┘           ││
                    │  │         Hybrid Fusion               ││
                    │  └─────────────────────────────────────┘│
                    │  ┌────────────┐  ┌─────────────────────┐│
                    │  │ FAISS IVF  │  │ Spectral Peak DB   ││
                    │  │ Vector DB  │  │ (Inverted Index)   ││
                    │  └────────────┘  └─────────────────────┘│
                    └─────────────────────────────────────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| Ingestion | ~2 min/episode (GPU) |
| Query Time | <500ms (CPU), <100ms (GPU) |
| Accuracy | ~95% on clean clips, ~85% on TikTok edits |
| Storage | ~10MB per episode |

## Tech Stack

- **Backend**: FastAPI, PyTorch, CLIP, FAISS, librosa
- **Frontend**: React, Vite
- **Deployment**: Docker, AWS ECS/EC2
