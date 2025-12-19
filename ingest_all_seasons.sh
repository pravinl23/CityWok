#!/bin/bash
# Ingest all seasons 1-15, one episode at a time
# Seasons are in /Users/pravinlohani/Downloads/Season X

# Usage: ingest_all_seasons.sh [start_season] [end_season]
# Example: ingest_all_seasons.sh 5 15  (start from season 5)

# Don't exit on error - we want to continue even if backend check fails
set +e

BASE_DIR="/Users/pravinlohani/Downloads"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

# Get optional start and end season parameters
START_SEASON=${1:-1}  # Default to 1 if not provided
END_SEASON=${2:-15}   # Default to 15 if not provided

cd "$BACKEND_DIR"
source venv/bin/activate

# Check if backend is already running
if curl -s http://localhost:8000/api/v1/test > /dev/null 2>&1; then
    echo "✅ Backend is already running!"
    BACKEND_PID=""
else
    echo "🚀 Starting backend server..."
    nohup uvicorn app.main:app --reload > backend.log 2>&1 &
    BACKEND_PID=$!
    
    echo "   Backend PID: $BACKEND_PID"
    echo "   Waiting for backend to start..."
    sleep 5
    
    # Check if backend started successfully
    if ! curl -s http://localhost:8000/api/v1/test > /dev/null 2>&1; then
        echo "⚠️  Backend may not have started. Check backend.log"
        echo "   Trying to continue anyway..."
    else
        echo "✅ Backend is running!"
    fi
fi
echo ""

echo "🎵 Starting ingestion of seasons ($START_SEASON-$END_SEASON)..."
echo "   Processing one episode at a time"
echo ""

# Process each season
for season in $(seq $START_SEASON $END_SEASON); do
    SEASON_DIR="$BASE_DIR/Season $season"
    
    if [ ! -d "$SEASON_DIR" ]; then
        echo "⚠️  Season $season not found at $SEASON_DIR, skipping..."
        continue
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📺 Processing Season $season"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Use the sequential ingestion script which processes one episode at a time
    python ingest_audio_sequential.py "$SEASON_DIR" "$season"
    
    echo ""
    echo "✅ Season $season complete!"
    echo ""
    
    # Delay between seasons (60 seconds to reduce heat)
    sleep 60
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All seasons ingested!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Check database status:"
echo "  cd backend && python check_db.py"
echo ""
if [ -n "$BACKEND_PID" ]; then
    echo "Backend is still running (PID: $BACKEND_PID)"
    echo "To stop backend: kill $BACKEND_PID"
else
    echo "Backend was already running (not started by this script)"
fi
echo ""

