import logging
import os

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT

log = logging.getLogger("uvicorn")


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
        description="Whether the file was found in the 'files/' directory.",
    )
    message: str = Field(
        ...,
        description="A message indicating the result of the operation.",
    )


def embed_file(request: EmbedRequest):
    """
    Checks if a file exists in the 'files/' directory.
    This is the first step in the embedding process.
    """
    file_path = PROJECT_ROOT / "files" / request.filename

    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": f"File not found: '{request.filename}' in 'files/' directory.",
            },
        )

    return {
        "success": True,
        "message": f"File '{request.filename}' is present.",
    }
