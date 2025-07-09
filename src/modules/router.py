from fastapi import APIRouter

from src.modules.ask.controller import router as ask_router
from src.modules.embed.controller import router as embed_router

# Main API router
api_router = APIRouter()

# Include module-specific routers
api_router.include_router(embed_router, prefix="/embed", tags=["embed"])
api_router.include_router(ask_router, prefix="/ask", tags=["ask"])
