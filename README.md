# CityWok 🍜

**Identify South Park episodes from audio/video clips using audio fingerprinting.**

Upload a TikTok, YouTube clip, or any audio/video file and CityWok will tell you which episode it's from and the exact timestamp.

## Features

- 🎵 **Audio Fingerprinting**: Shazam-style spectral peak landmark matching
- 🔄 **Auto-Conversion**: All files automatically converted to MP3 for consistent processing
- 🌐 **URL Support**: Direct TikTok, YouTube, and other platform URL support
- 🚀 **Fast & Accurate**: Optimized hash matching with early termination
- 📦 **Per-Season Databases**: Each season gets its own database file for faster ingestion

## Initial Setup

### 1. Install Backend Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Running the Application

### Start Backend Server

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

Backend will run on `http://localhost:8000`

### Start Frontend (Optional - for Web UI)

In a new terminal:

```bash
cd frontend
npm run dev
```

Frontend will run on `http://localhost:5173`

## Ingesting Episodes

Before you can identify clips, you need to build the audio fingerprint database by ingesting your South Park episodes.

### Prerequisites

Your episode files should be organized like:
```
/Users/pravinlohani/Downloads/
├── Season 1/
│   ├── S01E01.mp4
│   ├── S01E02.mp4
│   └── ...
├── Season 2/
│   ├── S02E01.mp4
│   └── ...
```

Supported formats: MP4, MOV, AVI, MKV, MPG, MPEG

### Configuration

By default, ingestion scripts look for episodes in `/Users/pravinlohani/Downloads/`.

To use a different location, set the `EPISODES_DIR` environment variable:

```bash
export EPISODES_DIR=/path/to/your/episodes
./backend/scripts/ingest_season.sh 6
```

### Ingest All Seasons (Recommended)

From the project root directory:

```bash
./backend/scripts/ingest_all_seasons.sh 1 20
```

This will:
- Automatically start the backend if needed
- Ingest seasons 1-20 sequentially
- Create separate database files: `audio_fingerprints_s01.pkl` through `s20.pkl`
- Wait 60 seconds between seasons to prevent overheating

### Ingest a Single Season

```bash
./backend/scripts/ingest_season.sh 6
```

This will ingest only Season 6 and create `audio_fingerprints_s06.pkl`

### Manual Ingestion (Alternative)

If you prefer to run the Python script directly:

```bash
cd backend
source venv/bin/activate
python scripts/ingestion/ingest_audio_sequential.py "/path/to/Season 1" 1
```

### Check Database Status

```bash
cd backend
python scripts/database/check_db.py
```

This will show you all ingested episodes and their fingerprint counts.

## API Endpoints

```
POST /api/v1/identify    - Identify episode from audio/video file or URL
POST /api/v1/ingest      - Add episode to database
GET  /api/v1/stats       - Database statistics
GET  /api/v1/test        - Health check
```

## Database Structure

CityWok uses per-season database files stored in `backend/data/`:
- `audio_fingerprints_s01.pkl` - Season 1
- `audio_fingerprints_s02.pkl` - Season 2
- ... (up to Season 20)

Each database contains audio fingerprint hashes for that season's episodes. This structure allows for faster ingestion and easier management.

## Tech Stack

- **Backend**: FastAPI, librosa, scipy, numpy, yt-dlp
- **Frontend**: React 19, Vite, Axios
- **Audio Processing**: Shazam-style spectral peak fingerprinting
- **Storage**: Pickle (per-season database files)
