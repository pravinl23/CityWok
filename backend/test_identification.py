#!/usr/bin/env python3
"""
Test script to verify episodes can be identified.
"""

import sys
import os
import requests
import tempfile
import cv2

def test_identification(episode_file: str, api_url: str = "http://localhost:8000"):
    """Test identifying an episode by creating a short clip from it."""
    if not os.path.exists(episode_file):
        print(f"Error: File not found: {episode_file}")
        return False
    
    print(f"Testing identification with: {os.path.basename(episode_file)}")
    print("Creating a 5-second test clip...")
    
    # Create a 5-second clip from the middle of the episode
    cap = cv2.VideoCapture(episode_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Start at 30 seconds (or middle if shorter)
    start_time = min(30, duration / 2)
    start_frame = int(start_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create temp file for clip
    temp_clip = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_clip_path = temp_clip.name
    temp_clip.close()
    
    # Write 5 seconds of video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_clip_path, fourcc, fps, 
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frames_to_write = int(5 * fps)
    for i in range(frames_to_write):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"Uploading clip for identification...")
    
    try:
        with open(temp_clip_path, 'rb') as f:
            files = {'file': ('test_clip.mp4', f, 'video/mp4')}
            response = requests.post(
                f"{api_url}/api/v1/identify",
                files=files,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Identification Result:")
            print(f"   Match found: {result.get('match_found', False)}")
            if result.get('match_found'):
                print(f"   Episode: {result.get('episode')}")
                print(f"   Timestamp: {result.get('timestamp')}")
                print(f"   Details: {result.get('details', {})}")
                return True
            else:
                print(f"   Message: {result.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_identification.py <episode_file>")
        sys.exit(1)
    
    test_identification(sys.argv[1])


