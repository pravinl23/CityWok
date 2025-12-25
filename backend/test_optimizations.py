#!/usr/bin/env python3
"""
Test script for performance optimizations:
1. Skip MP3 conversion (use extract_audio_to_memory)
2. Parallelize hash matching
"""

import os
import sys
import time
import requests

# Test with a local file or URL
if len(sys.argv) < 2:
    print("Usage: python test_optimizations.py <file_path_or_url>")
    print("Example: python test_optimizations.py https://www.tiktok.com/@southparkepisodes_/video/7199431355393051950")
    sys.exit(1)

test_input = sys.argv[1]
api_url = "http://localhost:8000"

print("="*60)
print("Testing Performance Optimizations")
print("="*60)
print(f"Input: {test_input}")
print(f"API: {api_url}")
print()

# Check if backend is running
try:
    response = requests.get(f"{api_url}/api/v1/test", timeout=3)
    if response.status_code != 200:
        print("❌ Backend is not responding correctly")
        sys.exit(1)
except Exception as e:
    print(f"❌ Backend is not running: {e}")
    print("   Start it with: uvicorn app.main:app --reload")
    sys.exit(1)

print("✓ Backend is running")
print()

# Make request
print("Sending request...")
start_time = time.time()

try:
    if test_input.startswith('http'):
        # URL request
        form_data = {'url': test_input}
        response = requests.post(
            f"{api_url}/api/v1/identify",
            data=form_data,
            timeout=300
        )
    else:
        # File upload
        with open(test_input, 'rb') as f:
            files = {'file': (os.path.basename(test_input), f)}
            response = requests.post(
                f"{api_url}/api/v1/identify",
                files=files,
                timeout=300
            )
    
    total_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print()
        print("="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total Time: {total_time:.2f}s")
        print()
        
        if result.get('match_found'):
            print(f"✓ Match Found!")
            print(f"  Episode: {result.get('episode')}")
            print(f"  Timestamp: {result.get('timestamp')}")
            print(f"  Confidence: {result.get('confidence')}%")
            print(f"  Aligned Matches: {result.get('aligned_matches')}")
            print(f"  Total Matches: {result.get('total_matches')}")
            if 'processing_time' in result:
                print(f"  Processing Time: {result['processing_time']:.2f}s")
        else:
            print("✗ No match found")
            print(f"  Message: {result.get('message')}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except requests.exceptions.Timeout:
    print(f"❌ Request timed out after 300 seconds")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)

