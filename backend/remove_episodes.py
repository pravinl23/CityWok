#!/usr/bin/env python3
"""
Script to remove specific episodes from the vector database.
Usage: python remove_episodes.py S02E01 S02E02 ...
"""

import sys
import os
import faiss
import json
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.services.vector_db import VectorDB

def remove_episodes(episode_ids: list):
    """Remove episodes from the vector database."""
    db = VectorDB()
    
    if db.index is None or db.index.ntotal == 0:
        print("Database is empty, nothing to remove.")
        return
    
    print(f"Current database size: {db.index.ntotal} vectors")
    
    # Find all indices to remove
    indices_to_remove = []
    for idx, metadata in db.metadata.items():
        ep_id = metadata.get('episode_id', '')
        if ep_id in episode_ids:
            indices_to_remove.append(idx)
    
    if not indices_to_remove:
        print(f"No matching episodes found to remove: {episode_ids}")
        return
    
    print(f"Found {len(indices_to_remove)} vectors to remove for episodes: {episode_ids}")
    
    # Create new index and metadata without the removed entries
    new_metadata = {}
    new_index = faiss.IndexFlatIP(db.dimension)
    
    removed_count = 0
    for idx in range(db.index.ntotal):
        if idx not in indices_to_remove:
            # Get the vector
            vector = db.index.reconstruct(idx)
            new_index.add(vector.reshape(1, -1))
            
            # Get the metadata
            if idx in db.metadata:
                new_metadata[new_index.ntotal - 1] = db.metadata[idx]
        else:
            removed_count += 1
    
    # Replace the old index and metadata
    db.index = new_index
    db.metadata = new_metadata
    
    # Save
    db.save_index()
    
    print(f"Removed {removed_count} vectors. New database size: {db.index.ntotal} vectors")
    print("Database updated successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_episodes.py S02E01 S02E02 ...")
        sys.exit(1)
    
    episode_ids = sys.argv[1:]
    remove_episodes(episode_ids)

