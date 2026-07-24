import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Use built-in sqlite3 (for macos where pysqlite3-binary not available)

import chromadb
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, MAX_EXAMPLES, CHROMA_DB_PATH
from .utils import normalize_text

import hashlib
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

# Persisted rows created before normalized retrieval kept raw-text embeddings.
# Bump this marker when the on-disk embedding schema changes again.
_SCHEMA_VERSION = "normalized-v1"
_SCHEMA_MARKER = os.path.join(CHROMA_DB_PATH, f".schema_{_SCHEMA_VERSION}")
_index_ready = False


def _example_id(message: str) -> str:
    """Stable, collision-resistant Chroma ID for a training example."""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _example_metadata(label: str, normalized: str) -> dict:
    return {"label": label, "norm_key": normalized}


def _sibling_ids_for_normalized(normalized: str) -> list[str]:
    """
    Find every persisted row that shares this normalized text.

    Prefer metadata.norm_key (post-migration). Only fall back to a full
    document scan when that metadata query finds nothing, so HITL writes stay
    O(matches) after migration instead of O(collection).
    """
    try:
        by_meta = collection.get(where={"norm_key": normalized}, include=[])
        meta_ids = list(dict.fromkeys(by_meta.get("ids") or []))
        if meta_ids:
            return meta_ids
    except Exception:
        pass

    try:
        rows = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    ids: list[str] = []
    for id_, doc, meta in zip(
        rows.get("ids") or [],
        rows.get("documents") or [],
        rows.get("metadatas") or [],
    ):
        if meta and meta.get("norm_key") == normalized:
            ids.append(id_)
            continue
        if doc is None:
            continue
        if (normalize_text(doc) or doc) == normalized:
            ids.append(id_)
    return list(dict.fromkeys(ids))


def _upsert_normalized_group(
    *,
    ids: list[str],
    normalized: str,
    label: str,
    embedding: list[float],
) -> None:
    if not ids:
        return
    meta = _example_metadata(label, normalized)
    collection.upsert(
        ids=ids,
        embeddings=[embedding] * len(ids),
        documents=[normalized] * len(ids),
        metadatas=[meta] * len(ids),
    )


def _reindex_all_normalized() -> bool:
    """
    Re-embed every row from its normalized document.

    Returns True when the migration finished successfully (including an empty
    collection). Returns False on recoverable storage failures so callers do
    not write the schema marker and permanently skip retry.
    """
    try:
        rows = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        logging.error("Failed to read Chroma examples for reindex: %s", e)
        return False

    ids = rows.get("ids") or []
    if not ids:
        logging.info(
            "No Chroma examples to reindex for normalized retrieval (%s)",
            _SCHEMA_VERSION,
        )
        return True

    updated = 0
    for id_, doc, meta in zip(
        ids,
        rows.get("documents") or [],
        rows.get("metadatas") or [],
    ):
        source = doc or ""
        normalized = normalize_text(source) or source
        if not normalized:
            continue
        label = (meta or {}).get("label", "SAFE")
        embedding = embedding_model.encode(normalized).tolist()
        collection.upsert(
            ids=[id_],
            embeddings=[embedding],
            documents=[normalized],
            metadatas=[_example_metadata(label, normalized)],
        )
        updated += 1

    logging.info(
        "Reindexed %s Chroma example(s) for normalized retrieval (%s)",
        updated,
        _SCHEMA_VERSION,
    )
    return True


def ensure_normalized_index() -> None:
    """
    One-time migration so persisted raw-vector rows match normalized queries.

    Safe to call repeatedly; uses an on-disk marker under CHROMA_DB_PATH.
    The marker is written only after a successful reindex.
    """
    global _index_ready
    if _index_ready:
        return
    if os.path.isfile(_SCHEMA_MARKER):
        _index_ready = True
        return
    try:
        if not _reindex_all_normalized():
            return
        with open(_SCHEMA_MARKER, "w", encoding="utf-8") as fh:
            fh.write(_SCHEMA_VERSION + "\n")
    except Exception as e:
        logging.error("Normalized embedding migration failed: %s", e)
        return
    _index_ready = True


def add_example(message: str, label: str) -> None:
  """
  Converts an example message into an embedding and stores it in ChromaDB.

  Args:
    message: The example text
    label: "SAFE" or "BAN"
  """
  ensure_normalized_index()

  # Keep the Chroma ID keyed to the raw HITL/seeds text so corrections replace
  # the same raw row. Reconcile every other row that shares the normalized
  # document (plain text or other Unicode variants) so RAG cannot keep
  # opposite labels for the same semantic example.
  original = message
  normalized = normalize_text(message) or message
  primary_id = _example_id(original)

  embedding = embedding_model.encode(normalized).tolist()
  sibling_ids = _sibling_ids_for_normalized(normalized)
  ids = list(dict.fromkeys([primary_id, *sibling_ids]))
  _upsert_normalized_group(
      ids=ids,
      normalized=normalized,
      label=label,
      embedding=embedding,
  )


def get_similar_examples(message: str) -> str:
  """
  Retrieves most similar examples from ChromaDB for a given input.

  Args:
    message: Incoming Discord message

  Returns:
    A formatted string of similar examples to inject into the prompt, or empty string if no relevant examples found.
  """
  ensure_normalized_index()

  if collection.count() == 0:
    return ""
  message = normalize_text(message) or message
  embedding = embedding_model.encode(message).tolist()
  results = collection.query(
    query_embeddings=[embedding],
    n_results=min(MAX_EXAMPLES, collection.count())
  )
  # Dedupe by normalized document. If legacy collisions still disagree on
  # label, omit that document rather than inject contradictory RAG context.
  chosen: dict[str, str] = {}
  conflicts: set[str] = set()
  for doc, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
  ):
    if distance >= SIMILARITY_THRESHOLD or not doc:
      continue
    label = (metadata or {}).get("label", "SAFE")
    if doc in conflicts:
      continue
    if doc in chosen and chosen[doc] != label:
      conflicts.add(doc)
      chosen.pop(doc, None)
      continue
    chosen[doc] = label

  examples = [
      f"Message: {doc}\nClassification: {label}"
      for doc, label in chosen.items()
  ]
  if not examples:
    return ""
  return "\n\n".join(examples)
