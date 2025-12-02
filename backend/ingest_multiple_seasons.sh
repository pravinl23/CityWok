#!/bin/bash
# Script to ingest multiple seasons in parallel
# Usage: ./ingest_multiple_seasons.sh <season1_dir> <season1_num> <season2_dir> <season2_num> ...

cd "$(dirname "$0")"
source venv/bin/activate

# Process arguments in pairs (directory, season_number)
while [ $# -ge 2 ]; do
    season_dir="$1"
    season_num="$2"
    shift 2
    
    echo "Starting ingestion for Season $season_num: $season_dir"
    python3 ingest_episodes.py "$season_dir" "$season_num" > "/tmp/ingest_s${season_num}.log" 2>&1 &
    echo "  → Running in background (PID: $!)"
    echo "  → Monitor with: tail -f /tmp/ingest_s${season_num}.log"
done

echo ""
echo "All seasons started in parallel. They will queue at the backend."
echo "Check status with: ps aux | grep ingest_episodes"
echo "Check database with: python3 check_database.py"

