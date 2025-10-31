import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.data_loader import load_all_faqs, prepare_faq_texts
from embeddings.embedder import FAQEmbedder
from embeddings.vector_store import FAISSVectorStore


def main():
    # Loading FAQ data and building FAISS index by extracting only the relevant fields for similarity search.
    print("=" * 60)
    print("Building FAISS Index for Tucows Knowledge Assistant")
    print("=" * 60)

    # Loading FAQ data
    print("\nLoading FAQ data...")
    faqs = load_all_faqs()
    texts = prepare_faq_texts(faqs)

    if not faqs:
        print("Error: No FAQs loaded. Check data directory.")
        return

    # Generating embeddings for all FAQ data using all-MiniLM-L6-v2 model to convert each text into a numerical vector of size 384 (the embedding dimension). Example: If we have 100 FAQs, we will get a (100, 384) NumPy array of embeddings.
    print("\nGenerating embeddings...")
    embedder = FAQEmbedder()
    embeddings = embedder.embed_texts(texts)

    # Initializing FAISS vector store and adding embeddings with metadata
    print("\nBuilding FAISS index...")
    vector_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    vector_store.add_vectors(embeddings, faqs)

    # Saving the FAISS index and metadata to disk for future use
    print("\nSaving index...")
    vector_store.save_index()

    print("\n" + "=" * 60)
    print("FAISS index has been created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()