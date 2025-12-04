#!/usr/bin/env python3
"""
Re-ingest all seasons with new spectral peak audio fingerprinting.
This clears the old audio database and re-processes all episodes.
"""

import os
import sys
import subprocess
import requests

def clear_audio_db():
    """Clear the old audio fingerprint database via API."""
    try:
        response = requests.post("http://localhost:8000/api/v1/admin/clear-audio", timeout=5)
        if response.status_code == 200:
            print("✓ Old audio database cleared via API")
            return True
        else:
            print(f"⚠️  API clear failed: {response.status_code}")
            # Fallback: delete file directly
            return clear_audio_db_file()
    except requests.exceptions.ConnectionError:
        print("⚠️  Backend not running. Clearing file directly...")
        return clear_audio_db_file()
    except Exception as e:
        print(f"⚠️  Error clearing via API: {e}. Clearing file directly...")
        return clear_audio_db_file()

def clear_audio_db_file():
    """Delete the audio database file directly."""
    from app.core.config import settings
    db_path = os.path.join(settings.DATA_DIR, "audio_fingerprints.pkl")
    if os.path.exists(db_path):
        print(f"🗑️  Deleting old audio database file: {db_path}")
        os.remove(db_path)
        print("✓ Old audio database file deleted")
        print("   (Restart backend to reload empty database)")
        return True
    else:
        print("ℹ️  No existing audio database found (will create new one)")
        return False

def reingest_season(season_path: str, season_num: int):
    """Re-ingest a single season with audio-only mode."""
    print(f"\n{'='*60}")
    print(f"📺 Season {season_num:02d}: {season_path}")
    print(f"{'='*60}\n")
    
    # Run the ingestion script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(script_dir, "ingest_episodes.py"),
        season_path,
        str(season_num),
        "--audio-only"
    ]
    
    result = subprocess.run(cmd, cwd=script_dir)
    return result.returncode == 0

if __name__ == "__main__":
    print("="*60)
    print("🎵 Re-Ingesting All Seasons with New Spectral Peak Audio")
    print("="*60)
    print("\nThis will:")
    print("  1. Clear old audio fingerprints")
    print("  2. Re-process all episodes with new Shazam-style algorithm")
    print("  3. Generate spectral peak fingerprints (much more accurate)")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    # Clear old database
    cleared = clear_audio_db()
    
    if cleared:
        print("\n⚠️  Note: The audio_matcher service may have cached the old DB.")
        print("   Restart the backend server after this completes for best results.")
        input("\nPress Enter to continue with re-ingestion...")
    
    # Base path for all seasons
    base_path = "/Users/pravinlohani/Downloads"
    
    successful_seasons = 0
    failed_seasons = 0
    
    # Process all seasons 1-15
    for season in range(1, 16):
        season_path = os.path.join(base_path, f"Season {season}")
        
        if not os.path.exists(season_path):
            print(f"⚠️  Season {season} path not found: {season_path}")
            continue
        
        try:
            if reingest_season(season_path, season):
                successful_seasons += 1
            else:
                failed_seasons += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print(f"Progress: Processed {successful_seasons} seasons successfully")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing Season {season}: {e}")
            import traceback
            traceback.print_exc()
            failed_seasons += 1
            continue
    
    print("\n" + "="*60)
    print(f"✅ Audio re-ingestion complete!")
    print(f"   Successful: {successful_seasons} seasons")
    print(f"   Failed: {failed_seasons} seasons")
    print("="*60)
    print("\n💡 Tip: Restart your backend server to reload the new audio database.")
