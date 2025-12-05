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

def ingest_episode(file_path: str, episode_id: str, api_url: str = "http://localhost:8000", show_progress: bool = True, audio_only: bool = False):
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
    response = None
    
    for url in urls_to_try:
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
                params = {'episode_id': episode_id}
                if audio_only:
                    params['audio_only'] = 'true'  # Skip video re-processing
                
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
                print(" ✗ Connection Failed")
                print(f"  Error: Could not connect to backend at {api_url}")
                print(f"  Details: {str(e)}")
                print("  Make sure the backend is running:")
                print("    cd backend && source venv/bin/activate && uvicorn app.main:app --reload")
                return False
            continue  # Try next URL
        except Exception as e:
            print(" ✗ Error")
            print(f"  Error: {e}")
            return False
    
    if response is None:
        print(" ✗ Connection Failed")
        print(f"  Error: Could not connect to backend")
        return False
            
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

def extract_episode_number(filename: str) -> tuple:
    """
    Extract season and episode numbers from filename.
    Returns (season_num, episode_num) or (None, None) if not found.
    """
    # Pattern 1: S##E## (e.g., S01E01, s2e5)
    match_sxe = re.search(r'[Ss](\d+)[Ee](\d+)', filename)
    if match_sxe:
        return (int(match_sxe.group(1)), int(match_sxe.group(2)))
    
    # Pattern 2: Episode X (e.g., Episode 1, episode 10)
    match_ep = re.search(r'Episode\s+(\d+)', filename, re.IGNORECASE)
    if match_ep:
        return (None, int(match_ep.group(1)))  # Season unknown
    
    # Pattern 3: Just a number at the start or end (e.g., "1.mp4", "Episode1.mp4")
    match_num = re.search(r'\b(\d+)\b', filename)
    if match_num:
        return (None, int(match_num.group(1)))
    
    return (None, None)

def sort_files_by_episode(files: list) -> list:
    """
    Sort files by episode number, handling numeric ordering correctly.
    """
    def get_sort_key(file_path):
        filename = file_path.stem
        season, episode = extract_episode_number(filename)
        if season is not None and episode is not None:
            return (season, episode)
        elif episode is not None:
            return (0, episode)  # Default to season 0 if unknown
        else:
            return (999, 999)  # Put unrecognized files at end
    
    return sorted(files, key=get_sort_key)

def ingest_directory(directory: str, api_url: str = "http://localhost:8000", skip_existing: bool = True, season: int = None, audio_only: bool = False):
    """
    Ingest all video files in a directory.
    
    Args:
        directory: Path to directory containing video files
        api_url: Backend API URL
        skip_existing: Whether to skip already-ingested episodes
        season: Season number to use for "Episode X" patterns (defaults to 1 if not specified)
    """
    directory = Path(directory)
    # Search recursively for video files in nested directories
    video_files = (
        list(directory.glob("**/*.mp4")) + 
        list(directory.glob("**/*.mov")) + 
        list(directory.glob("**/*.avi")) +
        list(directory.glob("**/*.mkv"))
    )
    
    if not video_files:
        print(f"No video files found in {directory}")
        return
    
    # Sort files by episode number (not alphabetically)
    video_files = sort_files_by_episode(video_files)
    
    # Check which episodes are already ingested
    already_ingested = set()
    if skip_existing:
        try:
            if audio_only:
                # For audio-only, check audio database instead
                from app.services.audio_fingerprint import audio_matcher
                import pickle
                from app.core.config import settings
                audio_db_path = os.path.join(settings.DATA_DIR, "audio_fingerprints.pkl")
                if os.path.exists(audio_db_path):
                    with open(audio_db_path, 'rb') as f:
                        fingerprints = pickle.load(f)
                    # Extract unique episode IDs from audio fingerprints
                    audio_episodes = set()
                    for ep_list in fingerprints.values():
                        for ep_id, _ in ep_list:
                            audio_episodes.add(ep_id)
                    already_ingested = audio_episodes
                    if already_ingested:
                        print(f"Found {len(already_ingested)} episodes with audio fingerprints: {sorted(already_ingested)}")
                else:
                    print("No audio fingerprint database found. Will process all episodes.")
            else:
                # Audio-only mode: check audio database
                from app.services.audio_fingerprint import audio_matcher
                already_ingested = set(audio_matcher.episode_hash_counts.keys()) if hasattr(audio_matcher, 'episode_hash_counts') else set()
                if already_ingested:
                    print(f"Found {len(already_ingested)} already ingested episodes: {sorted(already_ingested)}")
        except Exception as e:
            print(f"Warning: Could not check existing episodes: {e}")
    
    # Default season to 1 if not specified
    if season is None:
        season = 1
    
    print(f"Found {len(video_files)} video files. Starting ingestion...")
    if season:
        print(f"Using Season {season:02d} for episodes without explicit season number.")
    print("=" * 60)
    
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, video_file in enumerate(video_files, 1):
        filename = video_file.stem  # filename without extension
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {os.path.basename(video_file)}")
        
        # Extract season and episode numbers
        season_num, episode_num = extract_episode_number(filename)
        
        if season_num is not None and episode_num is not None:
            # Full S##E## pattern found
            episode_id = f"S{season_num:02d}E{episode_num:02d}"
        elif episode_num is not None:
            # Only episode number found, use provided/default season
            episode_id = f"S{season:02d}E{episode_num:02d}"
        else:
            # No pattern found - use filename
            episode_id = filename
            print(f"  Warning: Could not extract episode number from '{filename}'")
            print(f"  Using filename as episode ID: '{episode_id}'")
            print(f"  Consider renaming file to include episode number (e.g., 'Episode 1.mp4')")
        
        # Skip if already ingested
        if skip_existing and episode_id in already_ingested:
            print(f"  ⏭️  Skipping {episode_id} (already ingested)")
            skipped += 1
            print("-" * 60)
            continue
        
        try:
            if ingest_episode(str(video_file), episode_id, api_url, audio_only=audio_only):
                successful += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print(f"Progress: {successful} successful, {failed} failed, {skipped} skipped so far")
            raise
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed += 1
        
        print("-" * 60)
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {successful} successful, {failed} failed, {skipped} skipped out of {len(video_files)} total")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python ingest_episodes.py <file> <episode_id>")
        print("  Directory:   python ingest_episodes.py <directory> [season_number]")
        print("  Multiple:    python ingest_episodes.py <dir1> <season1> <dir2> <season2> ...")
        print()
        print("Examples:")
        print('  python ingest_episodes.py "S01E01.mp4" "S01E01"')
        print('  python ingest_episodes.py "/path/to/episodes/"')
        print('  python ingest_episodes.py "/path/to/episodes/" 2  # Specify Season 2')
        print('  python ingest_episodes.py "/path/s1/" 1 "/path/s2/" 2  # Multiple seasons')
        sys.exit(1)
    
    # Check if we have multiple directory/season pairs
    args = sys.argv[1:]
    if len(args) >= 4 and all(os.path.isdir(args[i]) for i in range(0, len(args), 2) if i < len(args)):
        # Multiple directories: process each pair
        print("=" * 60)
        print("Processing Multiple Seasons")
        print("=" * 60)
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                path = args[i]
                try:
                    season = int(args[i + 1])
                    print(f"\n[{i//2 + 1}] Processing Season {season}: {path}")
                    ingest_directory(path, season=season)
                except ValueError:
                    print(f"Warning: Invalid season number '{args[i + 1]}', skipping")
        sys.exit(0)
    
    # Check for --audio-only flag
    audio_only = '--audio-only' in sys.argv
    if audio_only:
        sys.argv.remove('--audio-only')
        print("=" * 60)
        print("AUDIO-ONLY MODE: Skipping video processing, only processing audio")
        print("=" * 60)
    
    # Single file or directory
    path = sys.argv[1]
    season = None
    
    if os.path.isdir(path):
        # Ingest all files in directory
        if len(sys.argv) >= 3:
            try:
                season = int(sys.argv[2])
                print(f"Using Season {season:02d} for episodes without explicit season number.")
            except ValueError:
                print(f"Warning: Invalid season number '{sys.argv[2]}', using default Season 1")
        ingest_directory(path, season=season, audio_only=audio_only)
    elif os.path.isfile(path):
        # Ingest single file
        if len(sys.argv) < 3:
            print("Error: Episode ID required for single file ingestion")
            print('Usage: python ingest_episodes.py <file> <episode_id>')
            sys.exit(1)
        episode_id = sys.argv[2]
        # Check for --audio-only flag
        audio_only = '--audio-only' in sys.argv
        success = ingest_episode(path, episode_id, audio_only=audio_only)
        sys.exit(0 if success else 1)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
