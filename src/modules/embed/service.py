import logging
from pathlib import Path
from typing import List, TypedDict

import numpy as np
from chromadb.api.models.Collection import Metadata
from chromadb.api.types import Embedding, OneOrMany
from pypdf import PdfReader

from src.config import settings
from src.services.chromadb import chroma_service
from src.services.openai import openai_service

log = logging.getLogger("uvicorn")


class Chunk(TypedDict):
    """Represents a text chunk with its metadata."""

    text: str
    page_number: int


class EmbeddedChunk(Chunk):
    """Represents a chunk with its embedding vector."""

    embedding: Embedding


def parse_and_chunk_pdf(file_path: Path) -> List[Chunk]:
    """
    Reads a PDF, extracts its text, and splits it into manageable chunks on a
    page-by-page basis. This is a simple and effective strategy for maintaining
    source context.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A list of text chunks, where each chunk contains the text and its page number.
    """
    log.info(f"Parsing and chunking PDF: {file_path}")
    reader = PdfReader(file_path)
    chunks: List[Chunk] = []

    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if not page_text:
            continue

        # Create overlapping chunks from the text of a single page
        for i in range(0, len(page_text), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
            chunk_text = page_text[i : i + settings.CHUNK_SIZE]
            chunks.append({"text": chunk_text, "page_number": page_num})

    log.info(f"Created {len(chunks)} chunks from {len(reader.pages)} pages.")
    return chunks


def create_embeddings_for_chunks(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    """
    Creates embeddings for a list of text chunks using a batch request.

    Args:
        chunks: A list of Chunks to be embedded.

    Returns:
        A list of EmbeddedChunks, with the embedding vector added.
    """
    log.info(f"Creating embeddings for {len(chunks)} chunks in a batch...")

    texts_to_embed = [chunk["text"] for chunk in chunks]
    if not texts_to_embed:
        return []

    embeddings = openai_service.create_embeddings(texts=texts_to_embed)

    embedded_chunks: List[EmbeddedChunk] = []
    for i, chunk in enumerate(chunks):
        embedded_chunks.append(
            {
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "embedding": np.array(embeddings[i], dtype=np.float32),
            }
        )

    log.info("Batch embeddings created successfully.")
    return embedded_chunks


def store_embeddings_in_chromadb(
    embedded_chunks: List[EmbeddedChunk],
    collection_name: str,
    document_name: str,
) -> None:
    """
    Stores embedded chunks in ChromaDB collection using batch operations.

    Args:
        embedded_chunks: A list of chunks with their embeddings.
        collection_name: The name of the ChromaDB collection.
        document_name: The name of the document being stored.
    """
    if not embedded_chunks:
        log.warning("No embedded chunks to store.")
        return

    log.info(f"Storing {len(embedded_chunks)} embeddings in ChromaDB collection '{collection_name}'...")

    # Get or create the collection
    collection = chroma_service.get_or_create_collection(name=collection_name)

    # Prepare data for batch insertion with proper typing
    ids: List[str] = [f"{document_name}_chunk_{i}" for i in range(len(embedded_chunks))]
    embeddings: OneOrMany[Embedding] = [chunk["embedding"].tolist() for chunk in embedded_chunks]
    documents: List[str] = [chunk["text"] for chunk in embedded_chunks]
    metadatas: OneOrMany[Metadata] = [
        {
            "document_name": document_name,
            "page_number": chunk["page_number"],
            "chunk_index": i,
        }
        for i, chunk in enumerate(embedded_chunks)
    ]

    # Add to ChromaDB in batch
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    log.info(f"Successfully stored {len(embedded_chunks)} embeddings in ChromaDB.")
