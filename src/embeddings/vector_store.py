# FAISS-based vector storage and retrieval for fast vector similarity search.
import json
import numpy as np
import faiss
from typing import List, Dict
from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH


class FAISSVectorStore:
    # FAISS index management for FAQ retrieval based on vector similarity.

    def __init__(self, embedding_dim: int = 384):
        # Initializing a FAISS index for inner product similarity search. Since our embeddings are normalized, inner product is the same as cosine similarity.
        self.embedding_dim = embedding_dim
        # Use IndexFlatIP for cosine similarity (Inner Product with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict] = []

    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict]):
        # Taking a list of embeddings and their corresponding metadata to add to the FAISS index.
        assert len(embeddings) == len(metadata), "Embeddings and metadata must match"

        # Converting to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')

        # Adding to FAISS index
        self.index.add(embeddings)
        self.metadata.extend(metadata)

        print(f"Added {len(embeddings)} vectors to FAISS index")
        print(f"Total vectors in index: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        # This function searches the FAISS index for the top_k most similar vectors to the query_embedding. It returns a list of metadata dictionaries for the most similar FAQs along with their similarity scores.

        # Reshaping for FAISS (since it expects a 2D array)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Searching (returns distances and indices)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):  # Valid index
                results.append({
                    'faq': self.metadata[idx],
                    'similarity_score': float(dist)  # Higher = more similar
                })

        return results

    def save_index(self):
        # This function saves the FAISS index and metadata to disk so that it can be reloaded later without rebuilding.
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        with open(FAISS_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
        print(f"Saved metadata to {FAISS_METADATA_PATH}")

    def load_index(self):
        # Loading FAISS index and metadata from disk for fast retrieval.
        if not FAISS_INDEX_PATH.exists() or not FAISS_METADATA_PATH.exists():
            raise FileNotFoundError("FAISS index files not found. Run build_index.py first.")

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(FAISS_METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
