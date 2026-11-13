"""Turns passages into vectors, stores them in a persistent Chroma collection,
reads a whole document back, and searches by meaning."""
import os
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import config


def _embedding_function():
    return SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME,
                                                device=config.EMBEDDING_DEVICE)


def _passage_number(chunk_id):
    found = re.search(r"-([0-9]+)$", chunk_id)
    return int(found.group(1)) if found else 0


def _require_passages(collection, document_name):
    stored = collection.get(where={"document_name": document_name}, include=["documents"])
    if not stored["ids"]:
        raise ValueError(config.INDEX_HOLDS_NO_CHUNKS_MESSAGE.format(document_name=document_name))
    return stored


def build_vector_index(chunk_records, store_path):
    """Build the searchable knowledge base from scratch; return the passages it holds."""
    os.makedirs(store_path, exist_ok=True)
    client = chromadb.PersistentClient(path=store_path)
    existing = [getattr(collection, "name", collection) for collection in client.list_collections()]
    if config.VECTOR_COLLECTION_NAME in existing:
        client.delete_collection(config.VECTOR_COLLECTION_NAME)
    collection = client.create_collection(
        name=config.VECTOR_COLLECTION_NAME, embedding_function=_embedding_function(),
        metadata={"hnsw:space": config.VECTOR_DISTANCE_METRIC})
    batch_size = 1000
    for start in range(0, len(chunk_records), batch_size):
        batch = chunk_records[start:start + batch_size]
        collection.add(ids=[chunk["chunk_id"] for chunk in batch],
                       documents=[chunk["chunk_text"] for chunk in batch],
                       metadatas=[{"document_name": chunk["document_name"]} for chunk in batch])
    return collection.count()


def open_vector_index(store_path):
    """Open the store indexing built, with the same embedding model, building nothing."""
    if not os.path.isfile(os.path.join(store_path, config.VECTOR_STORE_FILE_NAME)):
        raise ValueError(f"No knowledge store at {store_path}: it holds no "
                         f"{config.VECTOR_STORE_FILE_NAME}; run run_knowledge_indexing.py")
    client = chromadb.PersistentClient(path=store_path)
    return client.get_collection(name=config.VECTOR_COLLECTION_NAME,
                                 embedding_function=_embedding_function())


def read_document_chunks(collection, document_name):
    """Every passage stored for one document, in passage order, and no other document's."""
    stored = _require_passages(collection, document_name)
    ordered = sorted(zip(stored["ids"], stored["documents"]),
                     key=lambda pair: _passage_number(pair[0]))
    return [text for _, text in ordered]


def find_nearest_passage(collection, question, document_name):
    """The one passage of one document closest in meaning to one question."""
    _require_passages(collection, document_name)
    found = collection.query(query_texts=[question], n_results=1,
                             where={"document_name": document_name}, include=["documents"])
    return found["documents"][0][0]


def search_documents(collection, query_texts):
    """For each question, the closest passages of the documents the chat may read."""
    found = collection.query(
        query_texts=list(query_texts), n_results=config.CHAT_RETRIEVAL_RESULT_COUNT,
        where={"document_name": {"$nin": list(config.CHAT_EXCLUDED_DOCUMENTS)}},
        include=["documents", "metadatas", "distances"])
    results = []
    for ids, texts, metadatas, distances in zip(found["ids"], found["documents"],
                                                found["metadatas"], found["distances"]):
        passages = [{"chunk_id": chunk_id, "chunk_text": text,
                     "document_name": metadata["document_name"], "distance": float(distance)}
                    for chunk_id, text, metadata, distance in zip(ids, texts, metadatas,
                                                                   distances)]
        results.append(sorted(passages, key=lambda passage: passage["distance"]))
    return results