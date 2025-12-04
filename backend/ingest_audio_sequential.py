#!/usr/bin/env python3
"""
Sequential audio ingestion script - processes one episode at a time to reduce heat.
This is slower but much safer for your laptop.
"""

import os
import sys
import time
from pathlib import Path
import re
import requests

def extract_episode_number(filename: str) -> tuple:
    """Extract season and episode numbers from filename."""
    match_sxe = re.search(r'[Ss](\d+)[Ee](\d+)', filename)
    if match_sxe:
        return (int(match_sxe.group(1)), int(match_sxe.group(2)))
    match_ep = re.search(r'Episode\s+(\d+)', filename, re.IGNORECASE)
    if match_ep:
        return (None, int(match_ep.group(1)))
    match_num = re.search(r'\b(\d+)\b', filename)
    if match_num:
        return (None, int(match_num.group(1)))
    return (None, None)

def sort_files_by_episode(files: list) -> list:
    """Sort files by episode number."""
    def get_sort_key(file_path):
        filename = file_path.stem
        season, episode = extract_episode_number(filename)
        if season is not None and episode is not None:
            return (season, episode)
        elif episode is not None:
            return (0, episode)
        else:
            return (999, 999)
    return sorted(files, key=get_sort_key)

def get_already_ingested_episodes():
    """Check which episodes already have audio fingerprints."""
    try:
        from app.services.audio_fingerprint import audio_matcher
        import pickle
        from app.core.config import settings
        audio_db_path = os.path.join(settings.DATA_DIR, "audio_fingerprints.pkl")
        if os.path.exists(audio_db_path):
            try:
                with open(audio_db_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and 'fingerprints' in data:
                    fingerprints = data['fingerprints']
                else:
                    fingerprints = data
                audio_episodes = set()
                for ep_list in fingerprints.values():
                    if isinstance(ep_list, list):
                        for entry in ep_list:
                            if isinstance(entry, tuple) and len(entry) >= 1:
                                ep_id = entry[0]
                                audio_episodes.add(ep_id)
                return audio_episodes
            except:
                # Corrupted file - return empty set
                return set()
    except:
        pass
    return set()

def ingest_episode(file_path: str, episode_id: str, api_url: str = "http://localhost:8000"):
    """Ingest a single episode via API."""
    # Import the function directly instead of calling subprocess
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ingest_episodes import ingest_episode as ingest_func
    return ingest_func(file_path, episode_id, api_url, show_progress=False, audio_only=True)

def ingest_season_sequential(season_path: str, season_num: int, start_from_episode: int = 1):
    """Ingest a season sequentially, one episode at a time."""
    print(f"\n{'='*60}")
    print(f"📺 Season {season_num:02d}: {season_path}")
    print(f"{'='*60}\n")
    
    # Find all video files (recursively for nested folders)
    directory = Path(season_path)
    video_files = (
        list(directory.glob("**/*.mp4")) + 
        list(directory.glob("**/*.mov")) + 
        list(directory.glob("**/*.avi")) +
        list(directory.glob("**/*.mkv"))
    )
    
    if not video_files:
        print(f"⚠️  No video files found in {season_path}")
        return 0, 0
    
    # Sort by episode number
    video_files = sort_files_by_episode(video_files)
    
    # Check already ingested
    already_ingested = get_already_ingested_episodes()
    
    print(f"Found {len(video_files)} video files")
    if already_ingested:
        print(f"Found {len(already_ingested)} episodes already in audio database")
    print()
    
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, video_file in enumerate(video_files, 1):
        filename = video_file.stem
        season_num_extracted, episode_num = extract_episode_number(filename)
        
        if season_num_extracted is not None and episode_num is not None:
            episode_id = f"S{season_num_extracted:02d}E{episode_num:02d}"
        elif episode_num is not None:
            episode_id = f"S{season_num:02d}E{episode_num:02d}"
        else:
            episode_id = filename
            print(f"⚠️  Warning: Could not extract episode number from '{filename}'")
        
        # Skip if already ingested
        if episode_id in already_ingested:
            print(f"[{idx}/{len(video_files)}] ⏭️  Skipping {episode_id} (already ingested)")
            skipped += 1
            continue
        
        # Skip if before start_from_episode
        if episode_num is not None and episode_num < start_from_episode:
            print(f"[{idx}/{len(video_files)}] ⏭️  Skipping {episode_id} (before start point)")
            skipped += 1
            continue
        
        print(f"[{idx}/{len(video_files)}] 🎵 Processing {episode_id}...")
        print(f"   File: {os.path.basename(video_file)}")
        
        try:
            success = ingest_episode(str(video_file), episode_id)
            if success:
                print(f"   ✅ Success!")
                successful += 1
                # Update already_ingested to avoid re-checking
                already_ingested.add(episode_id)
            else:
                print(f"   ❌ Failed")
                failed += 1
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"Progress: {successful} successful, {failed} failed, {skipped} skipped")
            raise
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
        
        # Small delay between episodes to let system cool down
        if idx < len(video_files):
            time.sleep(2)
        
        print()
    
    return successful, failed

if __name__ == "__main__":
    print("="*60)
    print("🎵 Sequential Audio Ingestion (One Episode at a Time)")
    print("="*60)
    print("\nThis script processes episodes sequentially to reduce heat.")
    print("It will be slower but safer for your laptop.\n")
    
    base_path = "/Users/pravinlohani/Downloads"
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/api/v1/test", timeout=2)
        if response.status_code != 200:
            print("⚠️  Warning: Backend might not be running properly")
    except:
        print("❌ Error: Backend is not running!")
        print("Please start it first:")
        print("  cd backend && source venv/bin/activate && uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Process all seasons 1-15 sequentially
    total_successful = 0
    total_failed = 0
    
    for season in range(1, 16):
        season_path = os.path.join(base_path, f"Season {season}")
        
        if not os.path.exists(season_path):
            print(f"⚠️  Season {season} path not found: {season_path}")
            continue
        
        try:
            successful, failed = ingest_season_sequential(season_path, season)
            total_successful += successful
            total_failed += failed
            
            # Longer delay between seasons
            if season < 15:
                print(f"\n⏸️  Pausing 5 seconds before next season...\n")
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print(f"Progress: {total_successful} successful, {total_failed} failed")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing Season {season}: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1
            continue
    
    print("\n" + "="*60)
    print(f"✅ Sequential audio ingestion complete!")
    print(f"   Successful: {total_successful} episodes")
    print(f"   Failed: {total_failed} episodes")
    print("="*60)

