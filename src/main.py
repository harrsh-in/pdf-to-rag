from fastapi import FastAPI

from src.config import settings
from src.modules.router import api_router

app = FastAPI(
    title="RAG MVP Demo",
    description="A simple API to demonstrate Retrieval-Augmented Generation.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"environment": settings.ENVIRONMENT, "message": "Welcome to the RAG MVP!"}


app.include_router(api_router)
