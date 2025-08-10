"""
Configuration file for the Law RAG Chatbot application
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Load environment variables from .env file if it exists
def load_dotenv():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load .env file
load_dotenv()

# Hugging Face Configuration
HF_TOKEN = os.getenv('HF_TOKEN')
HF_DATASET_NAME = "ymoslem/Law-StackExchange"

# Groq Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = "llama3-8b-8192"  # or "mixtral-8x7b-32768"

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "law_stackexchange"

# FastAPI Configuration
API_TITLE = "Law RAG Chatbot API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "RAG-based legal assistance chatbot using Law-StackExchange data"
HOST = "0.0.0.0"
PORT = 8000

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 8  # Increased from 5
MAX_TOKENS = 4096
TEMPERATURE = 0.1
DEFAULT_CONTEXT_LENGTH = 5  # New default context length

# Dataset Configuration
DATASET_SPLIT = "train"
CACHE_DIR = ".cache"

# Error Messages
ERROR_MESSAGES = {
    "no_hf_token": "Hugging Face token not found. Set HF_TOKEN environment variable.",
    "no_groq_key": "Groq API key not found. Set GROQ_API_KEY environment variable.",
    "auth_failed": "Authentication failed: {}",
    "dataset_load_failed": "Failed to load dataset: {}",
    "embedding_failed": "Failed to create embeddings: {}",
    "vector_db_failed": "Failed to setup vector database: {}",
    "llm_failed": "Failed to initialize LLM: {}"
}

# API Response Models
class ChatRequest:
    question: str
    context_length: Optional[int] = 3

class ChatResponse:
    answer: str
    sources: list
    confidence: float
    processing_time: float 