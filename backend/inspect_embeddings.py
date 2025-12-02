#!/usr/bin/env python3
"""
Script to inspect embeddings in detail.
Usage: 
  python inspect_embeddings.py                    # Show all episodes
  python inspect_embeddings.py S02E01             # Show details for specific episode
  python inspect_embeddings.py S02E01 --sample 5  # Show sample embeddings
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_db import vector_db
from collections import defaultdict

def inspect_episode(episode_id: str = None, sample: int = 0):
    """Inspect embeddings for a specific episode or all episodes."""
    print("=" * 60)
    print("Embedding Inspection")
    print("=" * 60)
    
    if vector_db.index is None or vector_db.index.ntotal == 0:
        print("❌ Database is empty.")
        return
    
    print(f"\nTotal vectors: {vector_db.index.ntotal}")
    print(f"Embedding dimension: {vector_db.dimension}")
    
    # Group by episode
    episode_vectors = defaultdict(list)
    for idx, metadata in vector_db.metadata.items():
        ep_id = metadata.get('episode_id', 'unknown')
        if episode_id is None or ep_id == episode_id:
            episode_vectors[ep_id].append((idx, metadata))
    
    if not episode_vectors:
        print(f"\n❌ No episodes found" + (f" matching '{episode_id}'" if episode_id else ""))
        return
    
    print(f"\n📊 Found {len(episode_vectors)} episode(s):")
    print("-" * 60)
    
    for ep_id in sorted(episode_vectors.keys()):
        vectors = episode_vectors[ep_id]
        print(f"\n{ep_id}: {len(vectors)} frames")
        
        if sample > 0:
            print(f"  Sample of {min(sample, len(vectors))} embeddings:")
            for i, (idx, meta) in enumerate(vectors[:sample]):
                # Get the actual vector
                vector = vector_db.index.reconstruct(int(idx))
                timestamp = meta.get('timestamp', 0)
                
                # Check vector properties
                norm = np.linalg.norm(vector)
                has_nan = np.any(np.isnan(vector))
                has_inf = np.any(np.isinf(vector))
                
                print(f"    [{i+1}] Index {idx}: timestamp={timestamp:.2f}s, norm={norm:.4f}, "
                      f"NaN={has_nan}, Inf={has_inf}")
                if has_nan or has_inf:
                    print(f"         ⚠️  WARNING: Invalid values detected!")
        
        # Show timestamp range
        timestamps = [meta.get('timestamp', 0) for _, meta in vectors]
        if timestamps:
            print(f"  Timestamp range: {min(timestamps):.2f}s - {max(timestamps):.2f}s")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    episode_id = None
    sample = 0
    
    if len(sys.argv) > 1:
        episode_id = sys.argv[1]
    
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        if idx + 1 < len(sys.argv):
            try:
                sample = int(sys.argv[idx + 1])
            except ValueError:
                print("Error: --sample requires a number")
                sys.exit(1)
    
    inspect_episode(episode_id, sample)

