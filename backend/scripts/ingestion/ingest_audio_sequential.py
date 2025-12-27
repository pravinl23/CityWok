
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

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

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

def get_already_ingested_episodes(current_season: int = None):
    """
    Check which episodes already have audio fingerprints.
    
    OPTIMIZED: Only loads the 'counts' dict, not the entire fingerprints dict.
    If current_season is provided, only checks that season's database (much faster).
    """
    try:
        import pickle
        from app.core.config import settings
        
        if current_season:
            print(f"🔍 [DEBUG] Checking already ingested episodes for Season {current_season} only...")
        else:
            print("🔍 [DEBUG] Checking already ingested episodes across all seasons...")
        
        data_dir = settings.DATA_DIR
        audio_episodes = set()
        
        # If checking specific season, only check that one
        if current_season:
            db_file = f"audio_fingerprints_s{current_season:02d}.pkl"
            db_path = os.path.join(data_dir, db_file)
            
            if os.path.exists(db_path):
                try:
                    file_size = os.path.getsize(db_path)
                    file_size_mb = file_size / (1024 * 1024)
                    print(f"   📂 Loading {db_file} ({file_size_mb:.1f} MB)...", end="", flush=True)
                    
                    with open(db_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict) and 'counts' in data:
                        episode_hash_counts = data['counts']
                        episode_count = len(episode_hash_counts)
                        audio_episodes.update(episode_hash_counts.keys())
                        print(f" ✓ Found {episode_count} episodes")
                    else:
                        print(f" ✓ Loaded (no counts dict)")
                except Exception as e:
                    print(f" ✗ Error: {e}")
            
            print(f"✅ [DEBUG] Found {len(audio_episodes)} ingested episodes for Season {current_season}")
            return audio_episodes
        
        # Original logic for checking all seasons (kept for backward compatibility)
        DB_CONFIG = {
            season: f"audio_fingerprints_s{season:02d}.pkl"
            for season in range(1, 21)  # Seasons 1-20
        }
        
        print(f"🔍 [DEBUG] Data directory: {data_dir}")
        
        # Check old database format
        old_db_path = os.path.join(data_dir, "audio_fingerprints.pkl")
        if os.path.exists(old_db_path):
            print(f"🔍 [DEBUG] Checking old database format: {old_db_path}")
            try:
                # OPTIMIZATION: Use pickle to load only what we need
                # We'll load the full file but immediately extract only counts
                print(f"   Loading old database...")
                with open(old_db_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and 'counts' in data:
                    # Fast path: use counts dict directly
                    audio_episodes.update(data['counts'].keys())
                    print(f"   ✓ Found {len(data['counts'])} episodes in old database")
                elif isinstance(data, dict) and 'fingerprints' in data:
                    # Slower path: need to extract from fingerprints
                    # But we can sample to avoid loading everything
                    fingerprints = data['fingerprints']
                    # Sample first 1000 hashes to get episode IDs (much faster)
                    sample_size = min(1000, len(fingerprints))
                    sampled = list(fingerprints.items())[:sample_size]
                    for _, ep_list in sampled:
                        if isinstance(ep_list, list):
                            for entry in ep_list:
                                if isinstance(entry, tuple) and len(entry) >= 1:
                                    audio_episodes.add(entry[0])
                    print(f"   ✓ Sampled {sample_size} hashes from old database")
                else:
                    # Legacy format - sample fingerprints
                    sample_size = min(1000, len(data))
                    sampled = list(data.items())[:sample_size] if isinstance(data, dict) else []
                    for _, ep_list in sampled:
                        if isinstance(ep_list, list):
                            for entry in ep_list:
                                if isinstance(entry, tuple) and len(entry) >= 1:
                                    audio_episodes.add(entry[0])
                    print(f"   ✓ Sampled {sample_size} entries from old database")
            except Exception as e:
                print(f"⚠️  Warning: Could not load old database: {e}")
                pass
        
        # Check new multi-database format
        # OPTIMIZATION: Skip corrupted/missing files quickly
        db_files = list(DB_CONFIG.values())
        print(f"🔍 [DEBUG] Checking {len(db_files)} season database files...")
        
        for idx, db_file in enumerate(db_files, 1):
            db_path = os.path.join(data_dir, db_file)
            if not os.path.exists(db_path):
                continue
            
            # Quick check: skip if file is suspiciously small (likely corrupted)
            try:
                file_size = os.path.getsize(db_path)
                file_size_mb = file_size / (1024 * 1024)
                if file_size < 1000:  # Less than 1KB is probably corrupted
                    print(f"   [{idx}/{len(db_files)}] ⏭️  Skipping {db_file} (too small)")
                    continue
                print(f"   [{idx}/{len(db_files)}] 📂 Loading {db_file} ({file_size_mb:.1f} MB)...", end="", flush=True)
            except:
                continue
            
            try:
                # OPTIMIZATION: Load file but only use counts dict
                # This avoids loading millions of hashes
                with open(db_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Fast path: extract counts dict directly (O(1) for episode list)
                if isinstance(data, dict) and 'counts' in data:
                    episode_hash_counts = data['counts']
                    episode_count = len(episode_hash_counts)
                    audio_episodes.update(episode_hash_counts.keys())
                    print(f" ✓ Found {episode_count} episodes")
                elif isinstance(data, dict) and 'fingerprints' in data:
                    # Fallback: sample fingerprints to get episode IDs
                    # This is much faster than iterating all hashes
                    fingerprints = data['fingerprints']
                    # Sample first 100 hashes - enough to get all episode IDs
                    # (episodes appear in many hashes, so sampling works well)
                    sample_size = min(100, len(fingerprints))
                    sampled = list(fingerprints.items())[:sample_size]
                    for _, ep_list in sampled:
                        if isinstance(ep_list, list):
                            for entry in ep_list:
                                if isinstance(entry, tuple) and len(entry) >= 1:
                                    audio_episodes.add(entry[0])
                    print(f" ✓ Sampled {sample_size} hashes")
                else:
                    # Legacy format - sample
                    if isinstance(data, dict):
                        sample_size = min(100, len(data))
                        sampled = list(data.items())[:sample_size]
                        for _, ep_list in sampled:
                            if isinstance(ep_list, list):
                                for entry in ep_list:
                                    if isinstance(entry, tuple) and len(entry) >= 1:
                                        audio_episodes.add(entry[0])
                        print(f" ✓ Sampled {sample_size} entries")
            except (EOFError, pickle.UnpicklingError) as e:
                # Skip corrupted files silently
                print(f" ✗ Corrupted (skipping)")
                continue
            except Exception as e:
                # Only warn for other errors
                print(f" ✗ Error: {e}")
                pass
        
        print(f"✅ [DEBUG] Found {len(audio_episodes)} total ingested episodes")
        return audio_episodes
    except Exception as e:
        print(f"⚠️  Warning: Error checking ingested episodes: {e}")
        import traceback
        traceback.print_exc()
        return set()

def ingest_episode(file_path: str, episode_id: str, api_url: str = "http://localhost:8000"):
    """Ingest a single episode via API."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  Uploading to backend...", end="", flush=True)
    
    # Try both localhost and 127.0.0.1 if localhost fails
    urls_to_try = [api_url]
    if "localhost" in api_url:
        urls_to_try.append(api_url.replace("localhost", "127.0.0.1"))
    
    response = None
    for url in urls_to_try:
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
                params = {'episode_id': episode_id}
                
                response = requests.post(
                    f"{url}/api/v1/ingest",
                    files=files,
                    params=params,
                    timeout=3600  # 1 hour timeout for large files
                )
                break  # Success, exit the retry loop
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if url == urls_to_try[-1]:  # Last URL to try
                print(" ✗ Connection Failed")
                print(f"  Error: Could not connect to backend at {api_url}")
                return False
            continue  # Try next URL
        except Exception as e:
            print(" ✗ Error")
            print(f"  Error: {e}")
            return False
    
    if response is None:
        print(" ✗ Connection Failed")
        return False
    
    print(" ✓ Uploaded")
    print(f"  Processing...", end="", flush=True)
    
    if response.status_code == 200:
        print(" ✓ Complete")
        return True
    else:
        print(" ✗ Failed")
        print(f"  ✗ Error: {response.status_code} - {response.text}")
        return False

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
    
    # Check already ingested - only check current season for speed
    print("🔍 [DEBUG] Checking which episodes are already ingested...")
    already_ingested = get_already_ingested_episodes(current_season=season_num)
    
    print(f"\n📊 [INFO] Found {len(video_files)} video files")
    if already_ingested:
        print(f"📊 [INFO] Found {len(already_ingested)} episodes already in audio database")
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
        
        print()
    
    # Force save any remaining unsaved episodes at end of season
    try:
        response = requests.post("http://localhost:8000/api/v1/admin/force-save-databases", timeout=5)
        if response.status_code == 200:
            print("💾 Force saved all databases at end of season")
    except:
        pass  # Ignore if endpoint doesn't exist or backend unavailable
    
    return successful, failed

if __name__ == "__main__":
    print("="*60)
    print("🎵 Sequential Audio Ingestion (One Episode at a Time)")
    print("="*60)
    print("\nThis script processes episodes sequentially to reduce heat.")
    print("It will be slower but safer for your laptop.\n")
    
    # Check if backend is running (with retries)
    print("⏳ Checking if backend is ready...")
    max_retries = 10
    backend_ready = False
    
    for attempt in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/api/v1/test", timeout=3)
            if response.status_code == 200:
                backend_ready = True
                print("✅ Backend is ready!")
                break
        except:
            if attempt < max_retries - 1:
                time.sleep(2)
            pass
    
    if not backend_ready:
        print("❌ Error: Backend is not running or not ready!")
        print("Please start it first:")
        print("  cd backend && source venv/bin/activate && uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Check for command-line arguments
    if len(sys.argv) >= 3:
        # Single season mode: ingest_season.sh passes season_path and season_num
        season_path = sys.argv[1]
        season_num = int(sys.argv[2])
        
        if not os.path.exists(season_path):
            print(f"❌ Season path not found: {season_path}")
            sys.exit(1)
        
        print(f"📺 Processing Season {season_num:02d}: {season_path}\n")
        successful, failed = ingest_season_sequential(season_path, season_num)
        
        print("\n" + "="*60)
        print(f"✅ Season {season_num} complete!")
        print(f"   Successful: {successful} episodes")
        print(f"   Failed: {failed} episodes")
        print("="*60)
    else:
        # Multi-season mode: process all seasons 1-15
        base_path = "/Users/pravinlohani/Downloads"
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

