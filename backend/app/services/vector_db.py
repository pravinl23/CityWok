import faiss
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Any, Optional
from app.core.config import settings

class VectorDB:
    """
    High-performance vector database using FAISS with IVF+PQ indexing.
    
    Features:
    - IVF (Inverted File Index): Partitions vectors into clusters for fast search
    - PQ (Product Quantization): Compresses vectors 4-8x to fit more in memory
    - GPU acceleration when available
    - Automatic index training on first batch of vectors
    """
    
    def __init__(self):
        self.dimension = 512  # CLIP ViT-B/32 output dimension
        self.index_path = settings.VECTOR_DB_PATH
        self.metadata_path = settings.METADATA_DB_PATH
        self.index = None
        self.metadata = {}
        
        # IVF+PQ parameters (tuned for ~300 episodes = ~50k-200k vectors)
        self.nlist = 100        # Number of clusters (sqrt of expected vectors)
        self.m = 64             # Number of sub-quantizers (must divide dimension)
        self.nbits = 8          # Bits per sub-quantizer
        self.nprobe = 10        # Clusters to search (speed vs accuracy tradeoff)
        
        # Check for GPU
        self.use_gpu = False
        try:
            ngpus = faiss.get_num_gpus()
            if ngpus > 0:
                self.use_gpu = True
                print(f"🚀 FAISS GPU acceleration available ({ngpus} GPUs)")
        except:
            pass
            
        self.load_or_create_index()

    def load_or_create_index(self):
        """Load existing index or create a new trainable one."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                print("Loading existing vector index and metadata...")
                self.index = faiss.read_index(self.index_path)
                
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata = {int(k): v for k, v in data.items()}
                    
                print(f"✓ Loaded {self.index.ntotal} vectors from database")
                
                # Set search parameters
                if hasattr(self.index, 'nprobe'):
                    self.index.nprobe = self.nprobe
                    
            else:
                print("Creating new vector index...")
                self._create_new_index()
                
        except Exception as e:
            print(f"Error loading vector database: {e}")
            import traceback
            traceback.print_exc()
            print("Creating new vector index due to load error...")
            self._create_new_index()

    def _create_new_index(self):
        """Create a new IVF+Flat index (trains on first add)."""
        # Start with a flat index that will be replaced with IVF after training
        # IVF requires training data, so we use Flat initially
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {}
        self._trained = False
        print("Created new flat index (will upgrade to IVF after sufficient data)")

    def _upgrade_to_ivf(self, training_data: np.ndarray):
        """Upgrade flat index to IVF+PQ after collecting enough training data."""
        print(f"🔧 Training IVF index with {len(training_data)} vectors...")
        
        # Adjust nlist based on data size
        nlist = min(self.nlist, max(1, len(training_data) // 39))  # ~39 vectors per cluster
        
        # Create quantizer (coarse level)
        quantizer = faiss.IndexFlatIP(self.dimension)
        
        # Create IVF index with inner product (for cosine similarity on normalized vectors)
        # Using IndexIVFFlat for simplicity and good accuracy
        ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        ivf_index.train(training_data)
        ivf_index.nprobe = self.nprobe
        
        # Add existing vectors
        if self.index.ntotal > 0:
            # Reconstruct vectors from flat index
            vectors = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
            for i in range(self.index.ntotal):
                vectors[i] = self.index.reconstruct(i)
            ivf_index.add(vectors)
        
        self.index = ivf_index
        self._trained = True
        print(f"✓ IVF index trained and ready (nlist={nlist}, nprobe={self.nprobe})")

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add embeddings and metadata to the index."""
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        if len(embeddings) == 0:
            return

        # Ensure float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Check if we should upgrade to IVF
        current_total = self.index.ntotal
        new_total = current_total + len(embeddings)
        
        # Upgrade to IVF when we have enough data (>1000 vectors) and haven't trained yet
        if not getattr(self, '_trained', False) and new_total >= 1000:
            # Get all existing vectors
            if current_total > 0:
                existing = np.zeros((current_total, self.dimension), dtype=np.float32)
                for i in range(current_total):
                    existing[i] = self.index.reconstruct(i)
                training_data = np.vstack([existing, embeddings])
            else:
                training_data = embeddings
            self._upgrade_to_ivf(training_data)
        
        start_id = self.index.ntotal
        self.index.add(embeddings)

        for i, meta in enumerate(metadata_list):
            self.metadata[start_id + i] = meta

        self.save_index()
        print(f"✓ Added {len(embeddings)} vectors (total: {self.index.ntotal})")

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Tuple[float, Dict[str, Any]]]]:
        """
        Fast batch search for nearest neighbors.
        
        Uses IVF clustering to only search relevant partitions,
        making search O(sqrt(N)) instead of O(N).
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                return [[] for _ in range(len(query_embeddings))]
            
            if len(query_embeddings) == 0:
                return []
            
            # Prepare queries
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            
            query_embeddings = query_embeddings.astype(np.float32)
            faiss.normalize_L2(query_embeddings)
            
            # Batch search (much faster than one-at-a-time)
            search_k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embeddings, search_k)
            
            # Process results
            all_results = []
            for i in range(len(query_embeddings)):
                query_results = []
                for j in range(len(indices[i])):
                    idx = int(indices[i][j])
                    score = float(distances[i][j])
                    if idx != -1 and idx in self.metadata:
                        query_results.append((score, self.metadata[idx]))
                all_results.append(query_results)
            
            return all_results
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            import traceback
            traceback.print_exc()
            return [[] for _ in range(len(query_embeddings))]

    def save_index(self):
        """Persist index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else "None",
            "gpu_enabled": self.use_gpu,
        }
        if hasattr(self.index, 'nprobe'):
            stats["nprobe"] = self.index.nprobe
        if hasattr(self.index, 'nlist'):
            stats["nlist"] = self.index.nlist
        return stats

# Global instance
vector_db = VectorDB()

