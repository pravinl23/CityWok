# CityWok - Episode Identifier

A "Shazam for South Park" - Upload a video clip and identify which episode and timestamp it's from.

## How It Works

**Two-Step Process:**

1. **Ingest Episodes** (One-time setup): Upload episode videos to build the search database
   - System extracts frames and computes visual embeddings (CLIP)
   - System generates audio fingerprints
   - Data is stored in vector database for fast searching

2. **Identify Clips** (User-facing): Upload a short clip to find the episode
   - System searches the database using visual similarity and audio matching
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
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Deactivate Virtual Environment

When you're done:
```bash
deactivate
```

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
# Or manually:
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

### Step 2: Ingest Episodes

You need to ingest episodes before you can identify clips. You have two options:

**Option A: Using the Web UI**
1. Open the frontend at `http://localhost:5173`
2. Click "Admin: Ingest Episode"
3. Enter episode ID (e.g., "S01E01")
4. Upload the episode video file

**Option B: Using the Python Script** (Recommended for bulk ingestion)

```bash
cd backend
source venv/bin/activate
pip install requests  # If not already installed

# Ingest a single episode
python ingest_episodes.py "S01E01.mp4" "S01E01"

# Ingest all episodes in a directory
python ingest_episodes.py "/path/to/episodes/"
```

The script will automatically extract episode IDs from filenames if they follow the pattern `S##E##` (e.g., "S01E01.mp4").

### Step 3: Identify Clips

1. Open the frontend at `http://localhost:5173`
2. Click "Identify Clip"
3. Upload a short video clip (MP4, MOV, or AVI)
4. Wait for the system to identify the episode and timestamp

**Note:** Ingestion can take a few minutes per episode (depending on video length and hardware). Identification is much faster (usually 1-3 seconds).

