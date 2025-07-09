import logging
from typing import List, TypedDict

from src.config import settings
from src.services.chromadb import chroma_service
from src.services.openai import openai_service

log = logging.getLogger("uvicorn")


def create_embedding_for_query(query: str) -> List[float]:
    """
    Creates an embedding for a single query string.

    Args:
        query: The user's question.

    Returns:
        A list of floats representing the embedding vector.
    """
    log.info(f"Creating embedding for query: '{query}'")

    # The create_embeddings method can handle a single query by passing it as a list.
    embeddings = openai_service.create_embeddings(texts=[query])

    log.info("Successfully created embedding for the query.")
    return embeddings[0]


class RelevantChunk(TypedDict):
    """Represents a relevant chunk with its metadata."""

    text: str
    page_number: int
    document_name: str
    distance: float


def find_relevant_chunks(query_embedding: List[float], collection_name: str) -> List[RelevantChunk]:
    """
    Finds relevant chunks in ChromaDB based on a query embedding.

    Args:
        query_embedding: The embedding vector of the user's query.
        collection_name: The name of the collection to search in.

    Returns:
        A list of relevant chunks, including their text, metadata, and distance score.
    """
    log.info(f"Finding relevant chunks in collection '{collection_name}'...")
    collection = chroma_service.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=settings.TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    relevant_chunks: List[RelevantChunk] = []

    if not results or not results.get("ids") or not results["ids"][0]:
        log.warning(f"No relevant chunks found in collection '{collection_name}'.")
        return relevant_chunks

    documents = results.get("documents")
    metadatas = results.get("metadatas")
    distances = results.get("distances")

    if not documents or not metadatas or not distances:
        return relevant_chunks

    for i, doc_text in enumerate(documents[0]):
        metadata = metadatas[0][i]
        if isinstance(metadata, dict):
            page_number = metadata.get("page_number")
            document_name = metadata.get("document_name")

            relevant_chunks.append(
                {
                    "text": doc_text,
                    "page_number": int(page_number) if page_number is not None else 0,
                    "document_name": str(document_name) if document_name is not None else "Unknown",
                    "distance": distances[0][i],
                }
            )
        else:
            log.warning(f"Metadata for chunk {i} is not a dictionary: {metadata}")

    log.info(f"Found {len(relevant_chunks)} relevant chunks.")
    return relevant_chunks
