#!/usr/bin/env python3
"""
Build Common Hash Stoplist (Pickle Mode)

This script builds a stoplist of the most common hashes across all seasons
for the pickle-based audio fingerprinting system.

Usage:
    python scripts/database/build_stoplist_pickle.py [--top-percent 0.05] [--min-occurrences 500]
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.audio_fingerprint import AudioFingerprinter


def main():
    parser = argparse.ArgumentParser(description='Build common hash stoplist for pickle mode')
    parser.add_argument(
        '--top-percent',
        type=float,
        default=0.05,
        help='Top X%% of hashes by frequency to include (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--min-occurrences',
        type=int,
        default=500,
        help='Minimum number of occurrences to be considered common (default: 500)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Common Hash Stoplist Builder (Pickle Mode)")
    print("=" * 60)
    print(f"Top percent: {args.top_percent * 100}%")
    print(f"Min occurrences: {args.min_occurrences}")
    print()
    
    # Initialize fingerprinter (eager loading to scan all data)
    matcher = AudioFingerprinter(lazy_load=False)
    
    if not matcher.loaded_seasons:
        print("❌ No season databases found!")
        return 1
    
    # Build stoplist
    count = matcher.build_stoplist(
        top_percent=args.top_percent,
        min_occurrences=args.min_occurrences
    )
    
    print()
    print("=" * 60)
    print(f"✓ Successfully built stoplist with {count:,} common hashes")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

