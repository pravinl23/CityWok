#!/usr/bin/env python3
"""
Script to check what episodes are stored in the database.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_db import vector_db
from collections import Counter

def check_database():
    """Check what episodes are in the database."""
    print("=" * 60)
    print("Database Status Check")
    print("=" * 60)
    
    # Check if index exists
    index_path = vector_db.index_path
    metadata_path = vector_db.metadata_path
    
    print(f"\nIndex file: {index_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Index exists: {os.path.exists(index_path)}")
    print(f"Metadata exists: {os.path.exists(metadata_path)}")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("\n❌ Database files not found. No episodes have been ingested yet.")
        return
    
    # Load and check
    print(f"\nTotal vectors in index: {vector_db.index.ntotal}")
    print(f"Total metadata entries: {len(vector_db.metadata)}")
    
    if vector_db.index.ntotal == 0:
        print("\n❌ Database is empty. No episodes have been ingested.")
        return
    
    # Count episodes
    episode_counts = Counter()
    for meta in vector_db.metadata.values():
        ep_id = meta.get('episode_id', 'unknown')
        episode_counts[ep_id] += 1
    
    print(f"\n📊 Episodes in database:")
    print("-" * 60)
    for ep_id, count in sorted(episode_counts.items()):
        print(f"  {ep_id}: {count} frames")
    
    print(f"\n✅ Total: {len(episode_counts)} episodes, {vector_db.index.ntotal} frames")
    print("=" * 60)

if __name__ == "__main__":
    check_database()

