# CityWok - Episode Identifier

A "Shazam for South Park" - Upload a video clip and identify which episode and timestamp it's from.

## Quick Start

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Usage

### Ingest Episodes

```bash
cd backend
source venv/bin/activate
python3 ingest_episodes.py "/path/to/Season 1" 1
```

### Identify Clips

1. Open `http://localhost:5173`
2. Drag and drop a video clip
3. Get episode and timestamp
