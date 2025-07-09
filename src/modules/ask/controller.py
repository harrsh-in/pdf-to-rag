import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.config import settings
from src.modules.ask.service import create_embedding_for_query, find_relevant_chunks

log = logging.getLogger("uvicorn")
router = APIRouter()


class AskRequest(BaseModel):
    """Request model for the ask endpoint."""

    query: str = Field(
        ...,
        description="The user's question to ask the document.",
        examples=["What are the main findings of the document?"],
    )


class AskResponse(BaseModel):
    """Response model for the ask endpoint."""

    success: bool = Field(
        ...,
        description="Indicates whether the operation was successful.",
    )
    answer: str = Field(
        ...,
        description="The answer to the user's query.",
    )


@router.post("/", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Receives a user's question, finds relevant context from the vector store,
    and generates an answer using an LLM.
    """
    try:
        # 1. Create an embedding for the user's query
        log.info(f"Creating embedding for query: '{request.query}'")
        query_embedding = create_embedding_for_query(request.query)

        # 2. Find relevant chunks from ChromaDB
        log.info("Finding relevant chunks...")
        relevant_chunks = find_relevant_chunks(
            query_embedding=query_embedding,
            collection_name=settings.CHROMA_COLLECTION_NAME,
        )

        if not relevant_chunks:
            return AskResponse(
                success=True,
                answer="No relevant information found in the documents for your query.",
            )

        # 3. Format the response
        log.info("Formatting the final answer...")
        formatted_answer = "Based on the provided documents, here are the most relevant pieces of information:\n\n"
        for chunk in relevant_chunks:
            formatted_answer += f"- **Source**: {chunk['document_name']}, **Page**: {chunk['page_number']}\n"
            formatted_answer += f'  - **Text**: "{chunk["text"]}"\n'
            formatted_answer += f"  - **Similarity Score**: {chunk['distance']:.4f}\n\n"

        return AskResponse(success=True, answer=formatted_answer)

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)
        return AskResponse(
            success=False,
            answer="An unexpected internal server error occurred.",
        )
