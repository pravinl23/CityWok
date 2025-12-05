# CityWok 🍜

**Identify South Park episodes from audio/video clips using audio fingerprinting.**

Upload a TikTok, YouTube clip, or any audio/video file and CityWok will tell you which episode it's from and the exact timestamp.

## Features

- 🎵 **Audio Fingerprinting**: Shazam-style spectral peak landmark matching
- 🔄 **Auto-Conversion**: All files automatically converted to MP3 for consistent processing
- 🌐 **URL Support**: Direct TikTok, YouTube, and other platform URL support
- 🚀 **Fast & Accurate**: Optimized hash matching with early termination

## Quick Start

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

# Single episode
python ingest_episodes.py "/path/to/S01E01.mp4" "S01E01"

# Entire season
python ingest_audio_sequential.py "/path/to/Season 1" 1

# Multiple seasons
python ingest_audio_sequential.py "/path/to/Season 1" 1 "/path/to/Season 2" 2
```

## API Endpoints

```
POST /api/v1/identify    - Identify episode from audio/video file or URL
POST /api/v1/ingest      - Add episode to database
GET  /api/v1/stats       - Database statistics
GET  /api/v1/test        - Health check
```

## Tech Stack

- **Backend**: FastAPI, librosa, scipy, numpy
- **Frontend**: React, Vite
- **Audio Processing**: Shazam-style spectral peak fingerprinting
