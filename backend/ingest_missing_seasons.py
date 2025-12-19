#!/usr/bin/env python3
"""
Ingest missing seasons that weren't fully processed.
This script will check which seasons are missing and ingest them.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest_audio_sequential import ingest_season_sequential, get_already_ingested_episodes
import pickle
import glob

def get_seasons_in_database():
    """Get all seasons that have at least one episode in the database."""
    try:
        from app.core.config import settings
        data_dir = settings.DATA_DIR
        
        episodes = set()
        db_files = glob.glob(os.path.join(data_dir, "audio_fingerprints*.pkl"))
        
        for db_file in db_files:
            if not os.path.exists(db_file):
                continue
            try:
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'counts' in data:
                        episodes.update(data['counts'].keys())
            except Exception as e:
                print(f"⚠️  Error loading {db_file}: {e}")
                continue
        
        # Extract season numbers
        seasons = set()
        for ep_id in episodes:
            if ep_id.startswith('S'):
                try:
                    season = int(ep_id[1:3])
                    seasons.add(season)
                except ValueError:
                    pass
        
        return sorted(seasons)
    except Exception as e:
        print(f"Error checking database: {e}")
        return []

def main():
    BASE_DIR = "/Users/pravinlohani/Downloads"
    
    print("🔍 Checking which seasons are in the database...")
    seasons_in_db = get_seasons_in_database()
    print(f"   Found seasons in DB: {seasons_in_db}")
    print()
    
    # Check which seasons exist on disk
    all_seasons = []
    for season in range(1, 21):
        season_dir = f"{BASE_DIR}/Season {season}"
        if os.path.exists(season_dir):
            all_seasons.append(season)
    
    print(f"📁 Found {len(all_seasons)} season folders on disk: {all_seasons}")
    print()
    
    # Find missing seasons
    missing_seasons = [s for s in all_seasons if s not in seasons_in_db]
    
    if not missing_seasons:
        print("✅ All seasons are already ingested!")
        return
    
    print(f"⚠️  Missing seasons: {missing_seasons}")
    print()
    
    # Auto-proceed (can add --confirm flag later if needed)
    print(f"🚀 Proceeding with ingesting {len(missing_seasons)} missing season(s)...")
    print()
    
    print()
    print("=" * 70)
    print("STARTING INGESTION OF MISSING SEASONS")
    print("=" * 70)
    print()
    
    # Check if backend is running
    import requests
    backend_url = "http://localhost:8000/api/v1/test"
    try:
        response = requests.get(backend_url, timeout=2)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("⚠️  Backend responded but with unexpected status")
    except Exception:
        print("❌ Backend is not running!")
        print("   Please start the backend first:")
        print("   cd backend && source venv/bin/activate && uvicorn app.main:app --reload")
        return
    
    print()
    
    # Ingest each missing season
    total_successful = 0
    total_failed = 0
    
    for season in missing_seasons:
        season_dir = f"{BASE_DIR}/Season {season}"
        
        if not os.path.exists(season_dir):
            print(f"⚠️  Season {season} folder not found, skipping...")
            continue
        
        print("=" * 70)
        print(f"📺 Processing Season {season}")
        print("=" * 70)
        print()
        
        successful, failed = ingest_season_sequential(season_dir, season)
        total_successful += successful
        total_failed += failed
        
        print()
        print(f"✅ Season {season} complete: {successful} successful, {failed} failed")
        print()
        
        # Cooldown between seasons
        if season != missing_seasons[-1]:  # Don't wait after last season
            print("⏳ Waiting 60 seconds before next season...")
            import time
            time.sleep(60)
            print()
    
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Total: {total_successful} successful, {total_failed} failed")
    print()
    print("Check database status:")
    print("  cd backend && python check_db.py")

if __name__ == "__main__":
    main()

