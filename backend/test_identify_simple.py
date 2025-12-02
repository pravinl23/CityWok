#!/usr/bin/env python3
"""
Simple test script to identify a clip and see what happens.
"""

import requests
import sys

def test_identify(file_path: str):
    print(f"Testing identification with: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.split('/')[-1], f, 'video/mp4')}
            print("Uploading file...")
            response = requests.post(
                'http://localhost:8000/api/v1/identify',
                files=files,
                timeout=300  # 5 minute timeout
            )
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"Result: {result}")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 5 minutes")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the backend is running!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_identify.py <video_file>")
        sys.exit(1)
    
    test_identify(sys.argv[1])

