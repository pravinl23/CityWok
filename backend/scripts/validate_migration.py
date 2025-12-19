#!/usr/bin/env python3
"""
Validate LMDB Migration

Validates that LMDB databases match the original pickle databases.
Samples random hashes and compares posting lists.

Usage:
    python scripts/validate_migration.py [DATA_DIR] [LMDB_DIR]
"""

import sys
import os
import pickle
import random
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.storage.lmdb_store import LMDBFingerprintStore


def md5_to_int(md5_str: str) -> int:
    """Convert MD5 hash string to 64-bit integer."""
    hash_bytes = bytes.fromhex(md5_str)
    return int.from_bytes(hash_bytes[:8], byteorder='little')


def validate_season(
    pickle_path: str,
    lmdb_dir: str,
    season: int,
    sample_size: int = 1000
) -> bool:
    """
    Validate a single season's migration.

    Args:
        pickle_path: Path to original pickle file
        lmdb_dir: Directory containing LMDB database
        season: Season number
        sample_size: Number of hashes to sample for validation

    Returns:
        True if validation passed, False otherwise
    """
    print(f"\nValidating Season {season:02d}...")
    print(f"  Pickle: {pickle_path}")
    print(f"  LMDB: {lmdb_dir}")

    # Load pickle
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)

    pickle_fps = pickle_data.get('fingerprints', pickle_data if isinstance(pickle_data, dict) else {})
    pickle_counts = pickle_data.get('counts', {})

    # Open LMDB
    store = LMDBFingerprintStore(str(lmdb_dir), season=season, readonly=True)

    # Validate hash count
    lmdb_stats = store.get_stats()
    lmdb_hash_count = lmdb_stats['unique_hashes']

    print(f"\n  Hash counts:")
    print(f"    Pickle: {len(pickle_fps):,}")
    print(f"    LMDB: {lmdb_hash_count:,}")

    if lmdb_hash_count != len(pickle_fps):
        print(f"  ❌ Hash count mismatch!")
        store.close()
        return False

    # Sample random hashes for validation
    sample_hashes = random.sample(
        list(pickle_fps.keys()),
        min(sample_size, len(pickle_fps))
    )

    print(f"\n  Validating {len(sample_hashes):,} random hashes...")

    mismatches = 0
    missing = 0

    for i, md5_str in enumerate(sample_hashes):
        # Convert to integer
        hash_int = md5_to_int(md5_str)

        # Get from LMDB
        lmdb_entries = store.get_hash(hash_int)

        if not lmdb_entries:
            missing += 1
            if missing <= 5:
                print(f"    ⚠️  Hash {md5_str} missing from LMDB")
            continue

        # Get from pickle
        pickle_entries = pickle_fps[md5_str]

        # Sort for comparison
        lmdb_sorted = sorted(lmdb_entries)
        pickle_sorted = sorted(pickle_entries)

        # Compare
        if lmdb_sorted != pickle_sorted:
            mismatches += 1
            if mismatches <= 5:
                print(f"    ⚠️  Hash {md5_str} mismatch:")
                print(f"       Pickle: {len(pickle_entries)} entries")
                print(f"       LMDB: {len(lmdb_entries)} entries")

        if (i + 1) % 200 == 0:
            print(f"    Progress: {i+1}/{len(sample_hashes)}")

    store.close()

    # Summary
    print(f"\n  Results:")
    print(f"    Sampled: {len(sample_hashes):,} hashes")
    print(f"    Missing: {missing}")
    print(f"    Mismatches: {mismatches}")

    if missing > 0 or mismatches > 0:
        print(f"\n  ❌ Validation FAILED!")
        return False
    else:
        print(f"\n  ✅ Validation PASSED!")
        return True


def main():
    """Main entry point."""
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    lmdb_dir = sys.argv[2] if len(sys.argv) > 2 else "data/fingerprints/v1"

    print("="*60)
    print("LMDB Migration Validation")
    print("="*60)

    passed = 0
    failed = 0

    for season in range(1, 21):
        pickle_file = Path(data_dir) / f"audio_fingerprints_s{season:02d}.pkl"
        season_lmdb = Path(lmdb_dir) / f"season_{season:02d}.lmdb"

        if not pickle_file.exists():
            continue

        if not season_lmdb.exists():
            print(f"\n⚠️  Season {season:02d}: LMDB not found, skipping")
            continue

        try:
            if validate_season(str(pickle_file), str(season_lmdb), season):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Season {season:02d}: Validation error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Final summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"\nPassed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0 and passed > 0:
        print("\n✓ All validations passed!")
        sys.exit(0)
    else:
        print("\n✗ Some validations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
