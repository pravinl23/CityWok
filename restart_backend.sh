#!/bin/bash
# Kill any process on port 8000 and restart backend

echo "Killing processes on port 8000..."
lsof -ti :8000 | xargs kill -9 2>/dev/null
sleep 2

echo "Starting backend..."
cd /Users/pravinlohani/Projects/CityWok/backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

