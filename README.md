# CityWok - Episode Identifier

A "Shazam for South Park" - Upload a video clip and identify which episode and timestamp it's from.

## How It Works

**Two-Step Process:**

1. **Ingest Episodes** (One-time setup): Upload episode videos to build the search database
   - System extracts frames and computes visual embeddings (CLIP)
   - Data is stored in FAISS vector database for fast searching

2. **Identify Clips** (User-facing): Upload a short clip to find the episode
   - System searches the database using visual similarity
   - Returns episode ID and approximate timestamp

## Backend Setup

### 1. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
```

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or use the helper script:
```bash
./start_backend.sh
```

The API will be available at `http://localhost:8000`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

### Step 1: Start the Backend

```bash
./start_backend.sh
```

### Step 2: Ingest Episodes

You need to ingest episodes before you can identify clips. Use the Python script for bulk ingestion:

```bash
cd backend
source venv/bin/activate

# Ingest a single season
python3 ingest_episodes.py "/path/to/Season 1" 1

# Ingest multiple seasons in parallel
./ingest_multiple_seasons.sh "/path/to/Season 7" 7 "/path/to/Season 8" 8
```

The script automatically extracts episode IDs from filenames if they follow patterns like:
- `S##E##` (e.g., "S01E01.mp4")
- `Episode X` (e.g., "Episode 1.mp4")
- Numeric patterns

**Ingestion Progress:**
The script shows detailed progress for each episode:
- Progress counter (e.g., [1/13])
- File name and size
- Upload status
- Processing status
- Success/failure for each episode
- Final summary

### Step 3: Identify Clips

1. Open the frontend at `http://localhost:5173`
2. Drag and drop a video clip or click to upload
3. Wait for the system to identify the episode and timestamp

**Note:** Ingestion can take a few minutes per episode. Identification is much faster (usually 1-3 seconds).

## Database Management

### Check Database Status

```bash
cd backend
source venv/bin/activate
python3 check_database.py
```

### Remove Episodes

```bash
python3 remove_episodes.py S02E01 S02E02
```

### Inspect Embeddings

```bash
python3 inspect_embeddings.py S02E01 --sample 5
```

## Technical Details

### Architecture

- **Backend**: FastAPI with CLIP (OpenAI) for visual embeddings
- **Vector Database**: FAISS (IndexFlatIP) for similarity search
- **Frontend**: React with drag-and-drop support
- **Storage**: Local filesystem (vector index + metadata JSON)

### Scaling

FAISS handles 300+ episodes easily:
- **~230,000 frames** for 300 episodes
- **~460 MB** total size
- **Search speed**: ~1-5ms per query
- **Memory**: Fits easily in RAM

No need for external services like Pinecone until you reach 1M+ vectors.

### Data Storage

- **Vector Database**: `backend/data/vector_index.faiss`
- **Metadata**: `backend/data/metadata.json`
- **Temporary Uploads**: `backend/data/uploads/` (auto-cleaned)

## Troubleshooting

### Backend Not Starting

1. Make sure virtual environment is activated
2. Check if port 8000 is already in use: `lsof -i :8000`
3. Use restart script: `./restart_backend.sh`

### Ingestion Fails

1. Verify backend is running: `curl http://localhost:8000/api/v1/test`
2. Check file format (MP4, MKV, AVI supported)
3. Ensure sufficient disk space for temporary files

### Identification Returns Wrong Episode

- Visual matching is more reliable than audio
- Audio fingerprinting is currently disabled due to false positives
- Ensure episodes are properly ingested with full frame extraction

## Development

### Running Tests

```bash
cd backend
source venv/bin/activate
pytest tests/
```

### Project Structure

```
CityWok/
├── backend/
│   ├── app/
│   │   ├── api/          # API endpoints
│   │   ├── core/         # Configuration
│   │   └── services/     # Video processing, vector DB, audio
│   ├── data/             # Vector database and metadata
│   └── ingest_episodes.py
├── frontend/
│   └── src/
│       ├── App.jsx       # Main React component
│       └── assets/       # Images and static files
└── README.md

```
