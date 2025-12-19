#!/bin/bash
# Ingest a single season
# Usage: ./ingest_season.sh <season_number>
# Example: ./ingest_season.sh 6

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

if [ ! -d "$SEASON_DIR" ]; then
    echo "❌ Season $SEASON not found at $SEASON_DIR"
    exit 1
fi

cd "$BACKEND_DIR"
source venv/bin/activate

# Check if backend is running
if ! curl -s http://localhost:8000/api/v1/test > /dev/null 2>&1; then
    echo "🚀 Starting backend server..."
    nohup uvicorn app.main:app --reload > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    echo "   Waiting for backend to start..."
    sleep 5
    echo "✅ Backend started!"
else
    echo "✅ Backend is already running"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📺 Processing Season $SEASON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python ingest_audio_sequential.py "$SEASON_DIR" "$SEASON"

echo ""
echo "✅ Season $SEASON complete!"

