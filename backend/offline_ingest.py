#!/usr/bin/env python3
"""
Offline LMDB Database Builder

Builds LMDB fingerprint databases offline (without API server).
Can be run locally or in CI/CD pipelines.

Usage:
    # Ingest a single season
    python offline_ingest.py /path/to/Season1 1

    # Upload all databases to S3/R2
    python offline_ingest.py --upload data/fingerprints/v1

Examples:
    python offline_ingest.py "/Users/pravinlohani/Downloads/Season 1" 1
    python offline_ingest.py --upload data/fingerprints/v1
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Tuple

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

from app.core.audio_utils import extract_audio_to_memory
from app.services.audio_fingerprint_lmdb import AudioFingerprinterLMDB
from app.core.storage import FingerprintStorage


def extract_episode_id(file_path: Path) -> str:
    """
    Extract episode ID from filename.

    Supports formats:
    - S01E05.mp4 -> S01E05
    - South.Park.S01E05.mp4 -> S01E05
    - 01x05.mp4 -> S01E05
    """
    filename = file_path.stem  # Without extension

    # Try SxxExx format
    match = re.search(r'[Ss](\d+)[Ee](\d+)', filename)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        return f"S{season:02d}E{episode:02d}"

    # Try NNxNN format
    match = re.search(r'(\d+)x(\d+)', filename)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        return f"S{season:02d}E{episode:02d}"

    # Fallback: use filename
    return filename


def sort_episode_files(files: List[Path]) -> List[Path]:
    """Sort episode files by season and episode number."""

    def get_sort_key(file_path: Path) -> Tuple[int, int]:
        episode_id = extract_episode_id(file_path)
        match = re.search(r'[Ss](\d+)[Ee](\d+)', episode_id)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (999, 999)  # Put unrecognized files at end

    return sorted(files, key=get_sort_key)


def ingest_season(
    season_path: str,
    season_num: int,
    output_dir: str = "data/fingerprints/v1"
):
    """
    Ingest a season's episodes offline.

    Args:
        season_path: Path to directory containing episode files
        season_num: Season number
        output_dir: Output directory for LMDB databases
    """
    print(f"\n{'='*60}")
    print(f"Building Season {season_num:02d} Database")
    print(f"{'='*60}\n")

    directory = Path(season_path)

    if not directory.exists():
        print(f"❌ Directory not found: {season_path}")
        return False

    # Find all video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.mpg', '.mpeg', '.webm']
    video_files = []

    for ext in video_extensions:
        video_files.extend(directory.glob(f"*{ext}"))
        video_files.extend(directory.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"❌ No video files found in {season_path}")
        return False

    # Sort files by episode number
    video_files = sort_episode_files(video_files)

    print(f"Found {len(video_files)} video files")
    print(f"Output: {output_dir}/season_{season_num:02d}.lmdb\n")

    # Initialize fingerprinter
    fingerprinter = AudioFingerprinterLMDB(use_lmdb=True)
    fingerprinter.fingerprint_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    failed = 0

    for idx, video_file in enumerate(video_files, 1):
        episode_id = extract_episode_id(video_file)

        print(f"[{idx}/{len(video_files)}] Processing {episode_id}...")
        print(f"   File: {video_file.name}")

        try:
            # Extract audio to memory (no temp files!)
            print(f"   Extracting audio...")
            audio_array, sr = extract_audio_to_memory(str(video_file), sr=22050)

            duration = len(audio_array) / sr
            print(f"   Duration: {duration:.1f}s")

            # Add to LMDB database
            print(f"   Fingerprinting...")
            fingerprinter.add_episode_array(episode_id, audio_array, sr)

            print(f"   ✓ Success!\n")
            successful += 1

        except Exception as e:
            print(f"   ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    # Close databases
    fingerprinter.close()

    print(f"\n{'='*60}")
    print(f"Season {season_num:02d} Summary")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}\n")

    return successful > 0


def upload_to_storage(db_dir: str):
    """
    Upload all LMDB databases to S3/R2.

    Args:
        db_dir: Directory containing LMDB databases
    """
    print(f"\n{'='*60}")
    print(f"Uploading Databases to S3/R2")
    print(f"{'='*60}\n")

    storage = FingerprintStorage()

    if not storage.available:
        print("❌ S3 client not available (check AWS credentials)")
        return False

    db_path = Path(db_dir)

    if not db_path.exists():
        print(f"❌ Database directory not found: {db_dir}")
        return False

    # Find all season databases
    season_dbs = sorted(db_path.glob("season_*.lmdb"))

    if not season_dbs:
        print(f"❌ No season databases found in {db_dir}")
        return False

    print(f"Found {len(season_dbs)} season databases\n")

    successful = 0
    failed = 0

    for season_db in season_dbs:
        # Extract season number
        match = re.search(r'season_(\d+)', season_db.name)
        if not match:
            print(f"⚠️  Skipping {season_db.name} (invalid name format)")
            continue

        season_num = int(match.group(1))

        try:
            if storage.upload_season_db(season_num, str(season_db), create_checksum=True):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Upload Summary")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}\n")

    return failed == 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Upload mode
    if sys.argv[1] == '--upload':
        db_dir = sys.argv[2] if len(sys.argv) > 2 else "data/fingerprints/v1"
        success = upload_to_storage(db_dir)
        sys.exit(0 if success else 1)

    # Ingest mode
    season_path = sys.argv[1]
    season_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/fingerprints/v1"

    success = ingest_season(season_path, season_num, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
