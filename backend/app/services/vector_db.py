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
        try:
            if self.index is None:
                print("Warning: Vector index is None")
                return [[] for _ in range(len(query_embeddings))]
                
            if self.index.ntotal == 0:
                print("Warning: Vector index is empty")
                return [[] for _ in range(len(query_embeddings))]
            
            if len(query_embeddings) == 0:
                return []
            
            print(f"Searching {len(query_embeddings)} queries in index with {self.index.ntotal} vectors...")
            
            # Ensure embeddings are normalized for cosine similarity
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            
            # Check embedding dimensions match
            if query_embeddings.shape[1] != self.dimension:
                print(f"Error: Query embedding dimension {query_embeddings.shape[1]} doesn't match index dimension {self.dimension}")
                return [[] for _ in range(len(query_embeddings))]
            
            # Normalize query embeddings
            try:
                norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                query_embeddings = query_embeddings / norms
            except Exception as e:
                print(f"Error normalizing embeddings: {e}")
                return [[] for _ in range(len(query_embeddings))]
            
            # Search one query at a time to avoid any memory/crash issues
            all_results = []
            
            for i in range(len(query_embeddings)):
                try:
                    # Get single query
                    single_query = query_embeddings[i:i+1].copy()
                    single_query = np.ascontiguousarray(single_query.astype(np.float32))
                    
                    # Validate
                    if np.any(np.isnan(single_query)) or np.any(np.isinf(single_query)):
                        print(f"Warning: Query {i} has NaN/Inf, skipping")
                        all_results.append([])
                        continue
                    
                    # Search with just 1 query
                    search_k = min(k, self.index.ntotal)
                    if search_k <= 0:
                        all_results.append([])
                        continue
                    
                    distances, indices = self.index.search(single_query, search_k)
                    
                    # Process results for this single query
                    query_results = []
                    for m in range(len(indices[0])):
                        try:
                            idx = int(indices[0][m])
                            score = float(distances[0][m])
                            if idx != -1 and idx >= 0 and idx in self.metadata:
                                query_results.append((score, self.metadata[idx]))
                        except (ValueError, KeyError, IndexError) as e:
                            continue
                    
                    all_results.append(query_results)
                    
                    if (i + 1) % 5 == 0:
                        print(f"Processed {i + 1}/{len(query_embeddings)} queries...")
                        
                except Exception as e:
                    print(f"Error searching query {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append([])
            
            print(f"Search completed, returning {len(all_results)} result sets")
            return all_results
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            import traceback
            traceback.print_exc()
            # Return empty results instead of crashing
            return [[] for _ in range(len(query_embeddings))]

    def save_index(self):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

# Global instance
vector_db = VectorDB()

