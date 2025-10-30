import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4"

# RAG Settings
TOP_K_RETRIEVAL = 3  # Number of FAQs to retrieve
CONFIDENCE_THRESHOLD = 0.6  # Below this → needs human review

# Vector Store Settings
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "faqs.index"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"
