from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class FAQEmbedder:
    """This class turns text into numeric vectors (embeddings) using a pre-trained Sentence Transformer model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Loading HuggingFace's Sentence Transformer model ("all-MiniLM-L6-v2" by default; this has 384 dimensions or features per text, and is fast).
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Batch processing all FAQs, converting each String from a List of Strings (of FAQ data) into a normalized unit length vector (for easier cosine similarity) and returning a NumPy array with shape (number_of_texts, embedding_dim).
        """
        print(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encoding user questions (Strings) as single queries into vectors, normalizing them the same way as the FAQs, and returning a vector (1D NumPy array) of shape (embedding_dim,).
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding
