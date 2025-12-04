#!/usr/bin/env python3
"""
Continue audio re-ingestion from Season 3 onwards.
Seasons 1-2 are already done.
"""

import os
import sys
import subprocess

def reingest_season(season_path: str, season_num: int):
    """Re-ingest a single season with audio-only mode."""
    print(f"\n{'='*60}")
    print(f"📺 Season {season_num:02d}: {season_path}")
    print(f"{'='*60}\n")
    
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
    print("🎵 Continuing Audio Re-Ingestion: Seasons 4-15")
    print("="*60)
    print("\nSeasons 1-3 are already complete.")
    print("Processing remaining seasons...\n")
    
    base_path = "/Users/pravinlohani/Downloads"
    successful_seasons = 0
    failed_seasons = 0
    
    # Process seasons 4-15
    for season in range(4, 16):
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
    print(f"   Successful: {successful_seasons} seasons (4-15)")
    print(f"   Failed: {failed_seasons} seasons")
    print("="*60)

