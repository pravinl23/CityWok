#!/usr/bin/env python3
"""
Script to check audio fingerprint database status.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from app.core.config import settings
from collections import Counter

def check_audio_database():
    """Check what episodes are in the audio database."""
    print("=" * 60)
    print("Audio Fingerprint Database Status")
    print("=" * 60)
    
    audio_db_path = os.path.join(settings.DATA_DIR, "audio_fingerprints.pkl")
    
    print(f"\nAudio DB file: {audio_db_path}")
    print(f"Audio DB exists: {os.path.exists(audio_db_path)}")
    
    if not os.path.exists(audio_db_path):
        print("\n❌ Audio fingerprint database not found. No episodes have been processed for audio yet.")
        print("\n" + "=" * 60)
        print("NEXT SEASON TO START")
        print("=" * 60)
        print(f"\n🎯 Next season to start: Season 01")
        print("=" * 60)
        return 1
    
    try:
        # Check file size first
        file_size = os.path.getsize(audio_db_path)
        if file_size == 0:
            print(f"\n⚠️  Audio database file is empty (0 bytes)")
            print("\n" + "=" * 60)
            print("NEXT SEASON TO START")
            print("=" * 60)
            print(f"\n🎯 Next season to start: Season 01")
            print("=" * 60)
            return 1
        
        with open(audio_db_path, 'rb') as f:
            try:
                data = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"\n⚠️  Audio database file is corrupted: {e}")
                print(f"File size: {file_size} bytes")
                print("\n" + "=" * 60)
                print("NEXT SEASON TO START")
                print("=" * 60)
                print(f"\n🎯 Next season to start: Season 01")
                print("=" * 60)
                return 1
        
        # Handle both old and new format
        if isinstance(data, dict) and 'fingerprints' in data:
            fingerprints = data['fingerprints']
        else:
            fingerprints = data
        
        # Extract unique episode IDs
        audio_episodes = set()
        for ep_list in fingerprints.values():
            if isinstance(ep_list, list):
                for entry in ep_list:
                    if isinstance(entry, tuple) and len(entry) >= 1:
                        ep_id = entry[0]  # First element is episode_id
                        audio_episodes.add(ep_id)
                    elif isinstance(entry, str):
                        audio_episodes.add(entry)
        
        print(f"\nTotal unique episodes with audio: {len(audio_episodes)}")
        
        if len(audio_episodes) == 0:
            print("\n❌ Audio database is empty.")
            print("\n" + "=" * 60)
            print("NEXT SEASON TO START")
            print("=" * 60)
            print(f"\n🎯 Next season to start: Season 01")
            print("=" * 60)
            return 1
        
        # Count by season
        seasons = Counter()
        for ep in audio_episodes:
            if ep.startswith('S') and len(ep) >= 6:
                try:
                    season = int(ep[1:3])
                    seasons[season] += 1
                except:
                    pass
        
        print(f"\n📊 Audio episodes by season:")
        print("-" * 60)
        for season in sorted(seasons.keys()):
            count = sum(1 for ep in audio_episodes if ep.startswith(f'S{season:02d}'))
            season_eps = sorted([ep for ep in audio_episodes if ep.startswith(f'S{season:02d}')])
            print(f"  Season {season:02d}: {count} episodes")
            # Optionally show episode list (commented out for brevity)
            # for ep in season_eps:
            #     print(f"    - {ep}")
        
        print(f"\n✅ Total: {len(seasons)} seasons, {len(audio_episodes)} episodes")
        
        # Determine next season
        print("\n" + "=" * 60)
        print("NEXT SEASON TO START")
        print("=" * 60)
        if seasons:
            last_season = max(seasons.keys())
            next_season = last_season + 1
            print(f"\n🎯 Next season to start: Season {next_season:02d}")
            print(f"Last completed season: Season {last_season:02d}")
        else:
            print(f"\n🎯 Next season to start: Season 01")
            print("No seasons have been ingested yet for audio.")
        print("=" * 60)
        
        return next_season if seasons else 1
        
    except Exception as e:
        print(f"❌ Error reading audio database: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    next_season = check_audio_database()


