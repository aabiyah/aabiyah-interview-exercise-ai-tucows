# python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = os.path.join(BASE_DIR, "static")
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# LLM Provider (openai or ollama)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG Settings
TOP_K_RETRIEVAL = 3
CONFIDENCE_THRESHOLD = 0.6

# Vector Store Settings
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "faqs.index"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"
