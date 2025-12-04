#!/bin/bash
# Script to ingest seasons 2-15 in parallel for audio-only processing

BASE_DIR="/Users/pravinlohani/Downloads"
BACKEND_DIR="/Users/pravinlohani/Projects/CityWok/backend"

# Check if backend is running
curl -s http://localhost:8000/api/v1/test > /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Backend is not running. Please start it first:"
    echo "  cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
    exit 1
fi

echo "============================================================"
echo "Starting parallel audio ingestion for Seasons 2-15"
echo "============================================================"
echo ""

cd "$BACKEND_DIR"
source venv/bin/activate

# Start all seasons in parallel
for season in {2..15}; do
    SEASON_DIR="${BASE_DIR}/Season ${season}"
    
    if [ ! -d "$SEASON_DIR" ]; then
        echo "⚠️  Season ${season} directory not found: ${SEASON_DIR}"
        continue
    fi
    
    LOG_FILE="/tmp/ingest_audio_s${season}.log"
    echo "Starting Season ${season}..."
    echo "  → Directory: ${SEASON_DIR}"
    echo "  → Log file: ${LOG_FILE}"
    echo "  → Running in background..."
    
    # Run ingestion in background with audio-only flag
    python3 ingest_episodes.py "$SEASON_DIR" "$season" --audio-only > "$LOG_FILE" 2>&1 &
    
    echo "  → PID: $!"
    echo ""
done

echo "============================================================"
echo "All seasons started in parallel!"
echo ""
echo "Monitor progress with:"
echo "  tail -f /tmp/ingest_audio_s*.log"
echo ""
echo "Check individual seasons:"
echo "  tail -f /tmp/ingest_audio_s2.log  # Season 2"
echo "  tail -f /tmp/ingest_audio_s3.log  # Season 3"
echo "  # ... etc"
echo ""
echo "Check running processes:"
echo "  ps aux | grep ingest_episodes"
echo ""
echo "Check audio database status:"
echo "  cd backend && python3 check_audio_db.py"
echo "============================================================"


