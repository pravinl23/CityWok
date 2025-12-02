import faiss
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Tuple, Any
from app.core.config import settings

class VectorDB:
    def __init__(self):
        self.dimension = 512  # CLIP ViT-B/32 output dimension
        self.index_path = settings.VECTOR_DB_PATH
        self.metadata_path = settings.METADATA_DB_PATH
        self.index = None
        self.metadata = {}  # int_id -> dict
        self.load_or_create_index()

    def load_or_create_index(self):
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                print("Loading existing vector index and metadata...")
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    # Convert string keys back to int
                    data = json.load(f)
                    self.metadata = {int(k): v for k, v in data.items()}
                print(f"Loaded {self.index.ntotal} vectors from database")
            else:
                print("Creating new vector index...")
                # using Inner Product (IP) for cosine similarity on normalized vectors
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = {}
        except Exception as e:
            print(f"Error loading vector database: {e}")
            import traceback
            traceback.print_exc()
            # Create new index if loading fails
            print("Creating new vector index due to load error...")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = {}

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Adds embeddings and their corresponding metadata to the index.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")

        start_id = self.index.ntotal
        self.index.add(embeddings)

        for i, meta in enumerate(metadata_list):
            self.metadata[start_id + i] = meta

        self.save_index()

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Tuple[float, Dict[str, Any]]]]:
        """
        Search for nearest neighbors for each query embedding.
        Returns a list of results per query vector.
        Each result is a list of (score, metadata).
        """
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]
            
        distances, indices = self.index.search(query_embeddings, k)
        
        results = []
        for i in range(len(query_embeddings)):
            query_results = []
            for j in range(k):
                idx = indices[i][j]
                score = distances[i][j]
                if idx != -1 and idx in self.metadata:
                    query_results.append((float(score), self.metadata[idx]))
            results.append(query_results)
        
        return results

    def save_index(self):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

# Global instance
vector_db = VectorDB()

