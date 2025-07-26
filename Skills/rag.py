"""
RAG (Retrieval-Augmented Generation) Skill for Jarvis
- Ingests and embeds text files or pasted text
- Stores embeddings in ChromaDB
- Retrieves relevant context for user queries
"""
import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client and collection
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "rag_db")
chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
COLLECTION_NAME = "jarvis_rag"
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Load embedding model (small, fast, extensible)
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)


def ingest_text(text: str, source: str = "manual") -> str:
    """
    Ingest and embed a block of text, storing it in the vector DB.
    Returns a confirmation string.
    """
    if not text.strip():
        return "No text provided."
    embedding = embedder.encode([text])[0].tolist()
    doc_id = f"{source}_{abs(hash(text))}"
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{"source": source}]
    )
    return f"Text ingested and embedded (source: {source})."

def ingest_file(filepath: str) -> str:
    """
    Ingest and embed a text file.
    """
    if not os.path.isfile(filepath):
        return f"File not found: {filepath}"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return ingest_text(text, source=os.path.basename(filepath))

def retrieve(query: str, top_k: int = 3) -> List[str]:
    """
    Retrieve the most relevant text chunks for a query.
    Returns a list of text snippets.
    """
    if not query.strip():
        return []
    query_emb = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    return results.get("documents", [[]])[0]

def rag_skill(user_input: str, conversation_history=None, **kwargs) -> str:
    """
    RAG skill: Retrieve relevant context for a query, or ingest text/files.
    Usage:
      - rag ingest <text>         # Ingest pasted text
      - rag ingest_file <path>    # Ingest a text file
      - rag <query>               # Retrieve relevant context for a query
    """
    parts = user_input.strip().split(maxsplit=2)
    if len(parts) >= 2 and parts[1] == "ingest":
        if len(parts) < 3:
            return "Usage: rag ingest <text>"
        return ingest_text(parts[2])
    elif len(parts) >= 2 and parts[1] == "ingest_file":
        if len(parts) < 3:
            return "Usage: rag ingest_file <filepath>"
        return ingest_file(parts[2])
    else:
        # Treat as retrieval
        context_snippets = retrieve(user_input)
        if not context_snippets:
            return "No relevant context found."
        return "\n---\n".join(context_snippets)

def register(jarvis):
    """Register the RAG skill with Jarvis."""
    jarvis.register_skill("rag", rag_skill)
