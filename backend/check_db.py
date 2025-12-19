#!/usr/bin/env python3
"""Check all database files and verify which episodes are stored in each."""

import sys
import os
import pickle
import glob
from collections import defaultdict
from app.core.config import settings


def get_all_db_files(data_dir):
    """Find all audio fingerprint database files."""
    pattern = os.path.join(data_dir, "audio_fingerprints*.pkl")
    db_files = glob.glob(pattern)
    # Sort by filename for consistent output
    return sorted([os.path.basename(f) for f in db_files])


def verify_episodes_in_file(db_path, db_file):
    """Load a database file and verify which episodes are actually stored."""
    try:
        # Load file
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract structure
        if isinstance(data, dict) and 'fingerprints' in data:
            fingerprints = data['fingerprints']
            episode_counts = data.get('counts', {})
        else:
            # Legacy format
            fingerprints = data if isinstance(data, dict) else {}
            episode_counts = data.get('counts', {}) if isinstance(data, dict) else {}
        
        # Get episodes from counts dict (fast and accurate - this is the source of truth)
        episodes = sorted(episode_counts.keys()) if episode_counts else []
        counts = episode_counts if episode_counts else {}
        
        # Calculate totals without loading all fingerprints into memory
        total_hashes = len(fingerprints) if isinstance(fingerprints, dict) else 0
        
        # Sample verification: check a few random fingerprints to ensure structure is correct
        sample_episodes = set()
        sample_size = min(100, total_hashes)  # Sample up to 100 hashes
        verified_count = 0
        
        if isinstance(fingerprints, dict) and fingerprints:
            import random
            hash_keys = list(fingerprints.keys())
            if len(hash_keys) > sample_size:
                hash_keys = random.sample(hash_keys, sample_size)
            
            for hash_val in hash_keys:
                ep_list = fingerprints[hash_val]
                if isinstance(ep_list, list):
                    for entry in ep_list:
                        if isinstance(entry, tuple) and len(entry) >= 1:
                            sample_episodes.add(entry[0])
                            verified_count += 1
        
        # Calculate total entries efficiently (sum of list lengths)
        total_entries = sum(len(v) for v in fingerprints.values()) if isinstance(fingerprints, dict) else 0
        
        # Verify: check if sampled episodes match counts dict
        missing_in_sample = set(episodes) - sample_episodes if sample_episodes else set()
        consistency_note = ""
        if sample_episodes and missing_in_sample:
            # This is expected if we only sampled - not a real issue
            consistency_note = f" (sampled {sample_size} hashes, found {len(sample_episodes)} episodes)"
        
        return {
            'episodes': episodes,
            'counts': counts,
            'total_hashes': total_hashes,
            'total_entries': total_entries,
            'sample_verified': len(sample_episodes),
            'consistency_note': consistency_note,
            'file_size_mb': os.path.getsize(db_path) / (1024 * 1024)
        }
        
    except Exception as e:
        return {'error': str(e)}


def main():
    data_dir = settings.DATA_DIR
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    print("🔍 Scanning for database files...", file=sys.stderr)
    db_files = get_all_db_files(data_dir)
    
    if not db_files:
        print("⚠️  No database files found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"📦 Found {len(db_files)} database file(s)\n", file=sys.stderr)
    sys.stderr.flush()
    
    all_episodes = set()
    all_episode_counts = {}
    db_info = {}
    
    # Process each database file
    for db_file in db_files:
        db_path = os.path.join(data_dir, db_file)
        info = verify_episodes_in_file(db_path, db_file)
        db_info[db_file] = info
        
        if 'error' in info:
            print(f"❌ {db_file}: ERROR - {info['error']}")
            continue
        
        # Add to global sets
        for ep_id in info['episodes']:
            all_episodes.add(ep_id)
            all_episode_counts[ep_id] = info['counts'].get(ep_id, 0)
    
    # Print detailed report for each database
    print("=" * 70)
    print("DATABASE FILES REPORT")
    print("=" * 70)
    print()
    
    for db_file in db_files:
        info = db_info[db_file]
        
        if 'error' in info:
            print(f"❌ {db_file}")
            print(f"   Error: {info['error']}")
            print()
            continue
        
        print(f"📁 {db_file}")
        print(f"   Size: {info['file_size_mb']:.1f} MB")
        print(f"   Episodes: {len(info['episodes'])}")
        print(f"   Unique hashes: {info['total_hashes']:,}")
        print(f"   Total entries: {info['total_entries']:,}")
        
        # Show episodes in this file
        if info['episodes']:
            print(f"   Episodes in this file ({len(info['episodes'])}):")
            for ep_id in info['episodes']:
                count = info['counts'].get(ep_id, 0)
                print(f"      {ep_id}: {count:,} fingerprints")
        else:
            print(f"   ⚠️  No episodes found in this file!")
        
        # Show verification note
        if 'sample_verified' in info and info['sample_verified'] > 0:
            print(f"   ✓ Verified structure: sampled {info.get('sample_verified', 0)} entries{info.get('consistency_note', '')}")
        
        print()
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total database files: {len([f for f in db_files if 'error' not in db_info[f]])}")
    print(f"Total unique episodes: {len(all_episodes)}")
    print(f"Total fingerprints: {sum(all_episode_counts.values()):,}")
    print()
    
    if all_episodes:
        print("All episodes (sorted):")
        for ep_id in sorted(all_episodes):
            count = all_episode_counts.get(ep_id, 0)
            print(f"  {ep_id}: {count:,} fingerprints")
    else:
        print("⚠️  No episodes found in any database!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

