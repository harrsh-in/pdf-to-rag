from fastapi import APIRouter

from src.modules.embed.controller import router as embed_router

# Create a master router
api_router = APIRouter()

# Include the embed router.
# The endpoint defined in the embed_router will be available at /embed
api_router.include_router(embed_router, prefix="/embed", tags=["embed"])
