import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Use built-in sqlite3 (for macos where pysqlite3-binary not available)

import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, MAX_EXAMPLES, CHROMA_DB_PATH

import os
import warnings
import logging
import transformers
from huggingface_hub import logging as hf_logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
hf_logging.set_verbosity_error()


# Initialise embedding model and ChromaDB
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="examples")

def add_example(message: str, label: str) -> None:
  """
  Converts an example message into an embedding and stores it in ChromaDB.

  Args:
    message: The example text
    label: "SAFE" or "BAN"
  """
  embedding = embedding_model.encode(message).tolist()
  collection.upsert(
    embeddings = [embedding],
    documents = [message],
    metadatas=[{"label": label}],
    ids=[str(hash(message))]
  )

def get_similar_examples(message:str) -> str:
  """
  Retrieves most similar examples from ChromaDB for a given input.

  Args:
    message: Incoming Discord message

  Returns:
    A formatted string of similar examples to inject into the prompt, or empty string if no relevant examples found.
  """
  if collection.count() == 0:
    return ""
  embedding = embedding_model.encode(message).tolist()
  results = collection.query(
    query_embeddings=[embedding],
    n_results=min(MAX_EXAMPLES, collection.count())
  )
  examples = []
  for doc, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
  ):
    if distance < SIMILARITY_THRESHOLD:
      examples.append(f"Message: {doc}\nClassification: {metadata['label']}")

  if not examples:
    return ""
  return "\n\n".join(examples)