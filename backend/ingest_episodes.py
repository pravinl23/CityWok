#!/usr/bin/env python3
"""
Helper script to ingest South Park episodes into the database.
Usage: python ingest_episodes.py <episode_file> <episode_id>
Example: python ingest_episodes.py "S01E01.mp4" "S01E01"
"""

import sys
import os
import requests
from pathlib import Path
import re

def ingest_episode(file_path: str, episode_id: str, api_url: str = "http://localhost:8000", show_progress: bool = True):
    """
    Ingest a single episode into the database.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"[{episode_id}] Starting ingestion...")
    print(f"  File: {os.path.basename(file_path)} ({file_size_mb:.1f} MB)")
    print(f"  Uploading to backend...", end="", flush=True)
    
    # Try both localhost and 127.0.0.1 if localhost fails
    urls_to_try = [api_url]
    if "localhost" in api_url:
        urls_to_try.append(api_url.replace("localhost", "127.0.0.1"))
    
    last_error = None
    for url in urls_to_try:
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
                params = {'episode_id': episode_id}
                
                # Use stream=True to show progress, but for simplicity we'll just show upload start
                response = requests.post(
                    f"{url}/api/v1/ingest",
                    files=files,
                    params=params,
                    timeout=3600  # 1 hour timeout for large files
                )
                break  # Success, exit the retry loop
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if url == urls_to_try[-1]:  # Last URL to try
                raise
            continue  # Try next URL
    else:
        # If we exhausted all URLs, raise the last error
        if last_error:
            raise last_error
            
        print(" ✓ Uploaded")
        print(f"  Processing (extracting frames, computing embeddings, fingerprinting audio)...", end="", flush=True)
            
        if response.status_code == 200:
            result = response.json()
            print(" ✓ Complete")
            print(f"  ✓ Successfully ingested {episode_id}")
            return True
        else:
            print(" ✗ Failed")
            print(f"  ✗ Error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(" ✗ Connection Failed")
        print(f"  Error: Could not connect to backend at {api_url}")
        print(f"  Details: {str(e)}")
        print("  Make sure the backend is running:")
        print("    cd backend && source venv/bin/activate && uvicorn app.main:app --reload")
        return False
    except requests.exceptions.Timeout:
        print(" ✗ Timeout")
        print("  Error: Request timed out. The file might be too large or processing is taking too long.")
        return False
    except Exception as e:
        print(" ✗ Error")
        print(f"  Error: {e}")
        return False

def ingest_directory(directory: str, api_url: str = "http://localhost:8000"):
    """
    Ingest all video files in a directory.
    Assumes filenames contain episode identifiers (e.g., S01E01.mp4) or "Episode X".
    """
    directory = Path(directory)
    video_files = list(directory.glob("*.mp4")) + list(directory.glob("*.mov")) + list(directory.glob("*.avi"))
    
    if not video_files:
        print(f"No video files found in {directory}")
        return
    
    print(f"Found {len(video_files)} video files. Starting ingestion...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for idx, video_file in enumerate(sorted(video_files), 1):
        filename = video_file.stem  # filename without extension
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {os.path.basename(video_file)}")
        
        # Look for pattern like S##E## in filename
        match_sxe = re.search(r'[Ss](\d+)[Ee](\d+)', filename)
        
        # Look for "Episode X" pattern
        match_ep = re.search(r'Episode\s+(\d+)', filename, re.IGNORECASE)
        
        if match_sxe:
            season = match_sxe.group(1).zfill(2)
            episode = match_sxe.group(2).zfill(2)
            episode_id = f"S{season}E{episode}"
        elif match_ep:
            # Assume Season 1 if just "Episode X" is found (since user said Season 1)
            # Or we could ask, but for now defaulting to S01
            episode_num = match_ep.group(1).zfill(2)
            episode_id = f"S01E{episode_num}"
        else:
            # Use filename as episode_id if pattern not found
            episode_id = filename
            print(f"  Warning: Could not extract episode ID from {filename}, using '{episode_id}'")
        
        if ingest_episode(str(video_file), episode_id, api_url):
            successful += 1
        else:
            failed += 1
        
        print("-" * 60)
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {successful} successful, {failed} failed out of {len(video_files)} total")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python ingest_episodes.py <file> <episode_id>")
        print("  Directory:   python ingest_episodes.py <directory>")
        print()
        print("Examples:")
        print('  python ingest_episodes.py "S01E01.mp4" "S01E01"')
        print('  python ingest_episodes.py "/path/to/episodes/"')
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        # Ingest all files in directory
        ingest_directory(path)
    elif os.path.isfile(path):
        # Ingest single file
        if len(sys.argv) < 3:
            print("Error: Episode ID required for single file ingestion")
            print('Usage: python ingest_episodes.py <file> <episode_id>')
            sys.exit(1)
        episode_id = sys.argv[2]
        success = ingest_episode(path, episode_id)
        sys.exit(0 if success else 1)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
