from fastapi import APIRouter

from src.modules.embed.controller import EmbedResponse, embed_file

api_router = APIRouter()

api_router.post("/embed", response_model=EmbedResponse)(embed_file)
