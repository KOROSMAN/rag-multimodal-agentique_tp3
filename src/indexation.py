"""Accès à la base vectorielle ChromaDB du projet."""
from pathlib import Path
import chromadb

ROOT = Path(__file__).resolve().parent.parent   # racine du projet
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "pdf_clip_vit_b32"


def ouvrir_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )