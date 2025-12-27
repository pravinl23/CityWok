#!/bin/bash
# Ingest a single season
# Usage: ./ingest_season.sh <season_number>
# Example: ./ingest_season.sh 6

# Don't exit on error - we want to handle it
# set -x  # Uncomment for full debug mode

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <season_number>"
    echo "Example: $0 6"
    exit 1
fi

SEASON=$1
BASE_DIR="/Users/pravinlohani/Downloads"
SEASON_DIR="$BASE_DIR/Season $SEASON"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo "🔍 [DEBUG] Checking season directory: $SEASON_DIR"
if [ ! -d "$SEASON_DIR" ]; then
    echo "❌ Season $SEASON not found at $SEASON_DIR"
    exit 1
fi

# Count video files (recursively searches subdirectories)
echo "🔍 [DEBUG] Counting video files in $SEASON_DIR (including subdirectories)..."
VIDEO_COUNT=$(find "$SEASON_DIR" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" \) | wc -l | tr -d ' ')
echo "📊 [INFO] Found $VIDEO_COUNT video files in Season $SEASON"

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "⚠️  [WARNING] No video files found! Listing directory contents:"
    ls -la "$SEASON_DIR" | head -20
    echo ""
    echo "   Checking subdirectories..."
    find "$SEASON_DIR" -type d -maxdepth 2 | head -5
    exit 1
fi

echo "🔍 [DEBUG] Changing to backend directory: $BACKEND_DIR"
cd "$BACKEND_DIR"
echo "🔍 [DEBUG] Activating virtual environment..."
source venv/bin/activate

# Check if backend is running
echo "🔍 [DEBUG] Checking if backend is running..."
if ! curl -s http://localhost:8000/api/v1/test > /dev/null 2>&1; then
    echo "🚀 Starting backend server with lazy loading (fast startup)..."
    export LAZY_LOAD_PICKLE=true
    nohup env LAZY_LOAD_PICKLE=true uvicorn app.main:app --reload > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    echo "   Waiting for backend to start..."
    sleep 5
    
    # Verify backend started
    if curl -s http://localhost:8000/api/v1/test > /dev/null 2>&1; then
        echo "✅ Backend started successfully!"
    else
        echo "⚠️  [WARNING] Backend may not have started. Check backend.log"
        echo "   Continuing anyway..."
    fi
else
    echo "⚠️  Backend is already running, but may not have lazy loading enabled"
    echo "🔍 [DEBUG] Killing existing backend to restart with lazy loading..."
    pkill -f "uvicorn app.main:app" || true
    sleep 2
    
    echo "🚀 Starting backend server with lazy loading (fast startup)..."
    export LAZY_LOAD_PICKLE=true
    nohup env LAZY_LOAD_PICKLE=true uvicorn app.main:app --reload > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    echo "   Waiting for backend to start..."
    sleep 5
    
    # Verify backend started
    if curl -s http://localhost:8000/api/v1/test > /dev/null 2>&1; then
        echo "✅ Backend restarted with lazy loading!"
    else
        echo "⚠️  [WARNING] Backend may not have started. Check backend.log"
        echo "   Continuing anyway..."
    fi
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📺 Processing Season $SEASON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 [DEBUG] Season directory: $SEASON_DIR"
echo "🔍 [DEBUG] Video files found: $VIDEO_COUNT"
echo "🔍 [DEBUG] Starting Python ingestion script..."
echo ""

# Run with explicit output and error handling (unbuffered for real-time output)
python -u ingest_audio_sequential.py "$SEASON_DIR" "$SEASON" 2>&1 | tee -a "ingest_season_${SEASON}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Season $SEASON complete! (exit code: $EXIT_CODE)"
else
    echo "❌ Season $SEASON failed! (exit code: $EXIT_CODE)"
    echo "📋 Check log file: ingest_season_${SEASON}.log"
    exit $EXIT_CODE
fi

