#!/bin/bash
# Start script for the backend server

cd "$(dirname "$0")"
cd backend

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "Starting CityWok backend server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

