import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, settings
from src.modules.embed.service import (
    create_embeddings_for_chunks,
    parse_and_chunk_pdf,
    store_embeddings_in_chromadb,
)

log = logging.getLogger("uvicorn")
router = APIRouter()


class EmbedRequest(BaseModel):
    """Request model for the embed endpoint."""

    filename: str = Field(
        ...,
        description="The name of the file to embed, located in the 'files/' directory.",
        examples=["my-document.pdf"],
    )


class EmbedResponse(BaseModel):
    success: bool = Field(
        ...,
        description="Indicates whether the operation was successful.",
    )
    message: str = Field(
        ...,
        description="A message indicating the result of the operation.",
    )


@router.post("/", response_model=EmbedResponse)
def embed_file(request: EmbedRequest):
    """
    Reads a file from the `files/` directory, splits it into chunks,
    creates embeddings for each chunk, and stores them in ChromaDB.

    This endpoint is intended for processing files that are already present
    on the server.
    """
    file_path = PROJECT_ROOT / "files" / request.filename

    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": f"File not found: '{request.filename}' in 'files/' directory.",
            },
        )

    try:
        log.info(f"Processing file: {file_path}")

        # Parse and chunk the PDF
        chunks = parse_and_chunk_pdf(file_path)

        # Create embeddings for chunks
        embedded_chunks = create_embeddings_for_chunks(chunks)

        # Store embeddings in ChromaDB
        store_embeddings_in_chromadb(
            embedded_chunks=embedded_chunks,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            document_name=request.filename,
        )

        return EmbedResponse(
            success=True,
            message=f"Successfully parsed, chunked, embedded, and stored '{request.filename}' in ChromaDB.",
        )
    except ValueError as e:
        log.error(f"An error occurred while embedding '{request.filename}': {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)},
        )
    except Exception as e:
        log.error(f"An unexpected error occurred while embedding '{request.filename}': {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An unexpected internal server error occurred.",
            },
        )
