#!/usr/bin/env python3
"""
Migrate Pickle Databases to LMDB

Converts existing pickle-based fingerprint databases to LMDB format with:
- xxhash64 keys (currently keeps MD5 for compatibility)
- Binary-encoded posting lists with zstd compression
- Episode ID interning
- Validation checksums

Usage:
    python scripts/migrate_pickle_to_lmdb.py [DATA_DIR] [OUTPUT_DIR]

    DATA_DIR: Directory containing pickle files (default: data/)
    OUTPUT_DIR: Directory for LMDB databases (default: data/fingerprints/v1/)
"""

import sys
import os
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.storage.lmdb_store import (
    LMDBFingerprintStore,
    encode_posting_list,
    decode_posting_list
)


def md5_to_int(md5_str: str) -> int:
    """
    Convert MD5 hash string to 64-bit integer.

    For migration, we'll use the first 8 bytes of the MD5 hash.
    Future fingerprints will use xxhash64 directly.
    """
    # Convert hex string to bytes
    hash_bytes = bytes.fromhex(md5_str)
    # Use first 8 bytes as uint64
    return int.from_bytes(hash_bytes[:8], byteorder='little')


def migrate_season(
    pickle_path: str,
    output_dir: str,
    season: int,
    keep_md5: bool = False
) -> Dict[str, Any]:
    """
    Migrate a single season's pickle database to LMDB.

    Args:
        pickle_path: Path to pickle file
        output_dir: Directory for LMDB database
        season: Season number
        keep_md5: If True, store MD5 hashes as keys. If False, convert to int.

    Returns:
        Migration statistics
    """
    print(f"\n{'='*60}")
    print(f"Migrating Season {season:02d}")
    print(f"{'='*60}\n")
    print(f"  Source: {pickle_path}")

    # Load pickle database
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    fingerprints = data.get('fingerprints', data if isinstance(data, dict) else {})
    counts = data.get('counts', {})

    print(f"  Loaded {len(fingerprints):,} unique hashes")
    print(f"  Loaded {len(counts):,} episodes")

    # Create LMDB database
    lmdb_dir = Path(output_dir) / f"season_{season:02d}.lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Target: {lmdb_dir}")

    # Create LMDB store
    store = LMDBFingerprintStore(
        str(lmdb_dir),
        season=season,
        readonly=False,
        map_size=500 * 1024 * 1024  # 500 MB
    )

    # Migrate fingerprints
    print(f"\n  Converting hashes...")

    converted_count = 0
    total_entries = 0
    errors = 0

    start_time = time.time()

    for md5_str, entries in fingerprints.items():
        try:
            if keep_md5:
                # Store MD5 as bytes (16 bytes)
                hash_bytes = bytes.fromhex(md5_str)
            else:
                # Convert MD5 to 64-bit integer
                hash_int = md5_to_int(md5_str)

            # Store posting list
            if keep_md5:
                # For MD5 mode, we need to modify the storage layer
                # For now, skip this and use integer mode
                pass
            else:
                store.put_hash(hash_int, entries)

            converted_count += 1
            total_entries += len(entries)

            if converted_count % 10000 == 0:
                elapsed = time.time() - start_time
                rate = converted_count / elapsed if elapsed > 0 else 0
                print(f"    Progress: {converted_count:,}/{len(fingerprints):,} hashes ({rate:.0f} hashes/sec)")

        except Exception as e:
            print(f"    Error converting hash {md5_str}: {e}")
            errors += 1
            if errors > 100:
                print(f"    Too many errors, aborting!")
                raise

    # Store metadata
    metadata = {
        "season": season,
        "unique_hashes": len(fingerprints),
        "total_entries": total_entries,
        "episodes": counts,
        "hash_format": "md5_as_int64",
        "migrated_at": time.time(),
        "migrated_from": pickle_path
    }

    store.put_metadata('info', metadata)

    # Close database
    store.close()

    # Get final sizes
    lmdb_size = sum(f.stat().st_size for f in lmdb_dir.glob("*"))
    pickle_size = Path(pickle_path).stat().st_size

    elapsed_time = time.time() - start_time

    print(f"\n  ✓ Migration complete!")
    print(f"    Time: {elapsed_time:.1f}s")
    print(f"    Pickle size: {pickle_size / 1024 / 1024:.1f} MB")
    print(f"    LMDB size: {lmdb_size / 1024 / 1024:.1f} MB")
    print(f"    Reduction: {(1 - lmdb_size/pickle_size)*100:.1f}%")
    print(f"    Errors: {errors}")

    return {
        "season": season,
        "unique_hashes": len(fingerprints),
        "total_entries": total_entries,
        "episodes": len(counts),
        "pickle_size": pickle_size,
        "lmdb_size": lmdb_size,
        "errors": errors,
        "elapsed_time": elapsed_time
    }


def migrate_all_seasons(
    data_dir: str,
    output_dir: str,
    seasons: range = range(1, 21)
):
    """
    Migrate all season pickle databases to LMDB.

    Args:
        data_dir: Directory containing pickle files
        output_dir: Directory for LMDB databases
        seasons: Range of season numbers to migrate
    """
    print("="*60)
    print("Pickle → LMDB Migration Tool")
    print("="*60)
    print(f"\nSource directory: {data_dir}")
    print(f"Output directory: {output_dir}\n")

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = []

    for season in seasons:
        pickle_file = data_path / f"audio_fingerprints_s{season:02d}.pkl"

        if not pickle_file.exists():
            print(f"⏭️  Season {season:02d}: File not found")
            skipped.append(season)
            continue

        try:
            result = migrate_season(
                str(pickle_file),
                str(output_path),
                season,
                keep_md5=False  # Convert to integer hashes
            )
            results.append(result)

        except Exception as e:
            print(f"\n❌ Season {season:02d}: Migration failed!")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)

    if results:
        total_pickle = sum(r['pickle_size'] for r in results)
        total_lmdb = sum(r['lmdb_size'] for r in results)
        total_hashes = sum(r['unique_hashes'] for r in results)
        total_entries = sum(r['total_entries'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        total_time = sum(r['elapsed_time'] for r in results)

        print(f"\nSeasons migrated: {len(results)}")
        print(f"Seasons skipped: {len(skipped)} {skipped if skipped else ''}")
        print(f"\nTotal unique hashes: {total_hashes:,}")
        print(f"Total entries: {total_entries:,}")
        print(f"Total pickle size: {total_pickle / 1024 / 1024 / 1024:.2f} GB")
        print(f"Total LMDB size: {total_lmdb / 1024 / 1024 / 1024:.2f} GB")
        print(f"Total reduction: {(1 - total_lmdb/total_pickle)*100:.1f}%")
        print(f"Total errors: {total_errors}")
        print(f"Total time: {total_time:.1f}s")

        print(f"\n✓ Migration complete!")
        print(f"  LMDB databases saved to: {output_dir}")

    else:
        print("\n❌ No seasons were successfully migrated")


def main():
    """Main entry point."""
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/fingerprints/v1"
    max_seasons = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    # Validate data directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Run migration (only first max_seasons)
    migrate_all_seasons(data_dir, output_dir, seasons=range(1, max_seasons + 1))


if __name__ == "__main__":
    main()
