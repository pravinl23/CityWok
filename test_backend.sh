#!/bin/bash
# Script to test the backend step by step

echo "Testing CityWok Backend..."
echo "============================"

echo ""
echo "1. Testing root endpoint..."
curl -s http://localhost:8000/ || echo "✗ Failed"

echo ""
echo "2. Testing test endpoint..."
curl -s http://localhost:8000/api/v1/test || echo "✗ Failed"

echo ""
echo "3. Testing identify endpoint with file..."
if [ -f "/Users/pravinlohani/Projects/CityWok/v09044g40000cfohp03c77u18jdukpng.MP4" ]; then
    curl -X POST -F "file=@/Users/pravinlohani/Projects/CityWok/v09044g40000cfohp03c77u18jdukpng.MP4" \
         http://localhost:8000/api/v1/identify \
         --max-time 180 \
         -v 2>&1 | head -50
else
    echo "✗ Test file not found"
fi

echo ""
echo "============================"
echo "Done"


