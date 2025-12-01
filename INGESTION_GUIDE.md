# Quick Start Guide

## To See Ingestion Progress:

1. **Make sure backend is running** (in one terminal):
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

2. **Run ingestion script** (in another terminal):
   ```bash
   cd backend
   source venv/bin/activate
   python3 ingest_episodes.py "/Users/pravinlohani/Downloads/Season 1"
   ```

The script will now show:
- Progress counter (e.g., [1/13])
- File name and size
- Upload status
- Processing status
- Success/failure for each episode
- Final summary

You'll see output like:
```
Found 13 video files. Starting ingestion...
============================================================

[1/13] Processing: Episode 1 (REMASTER).mp4
[S01E01] Starting ingestion...
  File: Episode 1 (REMASTER).mp4 (250.5 MB)
  Uploading to backend... ✓ Uploaded
  Processing (extracting frames, computing embeddings, fingerprinting audio)... ✓ Complete
  ✓ Successfully ingested S01E01
------------------------------------------------------------
```

