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
        return
    
    try:
        with open(audio_db_path, 'rb') as f:
            fingerprints = pickle.load(f)
        
        # Extract unique episode IDs
        audio_episodes = set()
        for ep_list in fingerprints.values():
            for ep_id, _ in ep_list:
                audio_episodes.add(ep_id)
        
        print(f"\nTotal unique episodes with audio: {len(audio_episodes)}")
        
        if len(audio_episodes) == 0:
            print("\n❌ Audio database is empty.")
            return
        
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
            print(f"  Season {season:02d}: {seasons[season]} episodes")
        
        print(f"\n✅ Total: {len(seasons)} seasons, {len(audio_episodes)} episodes")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error reading audio database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_audio_database()

