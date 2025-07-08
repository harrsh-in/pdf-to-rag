from fastapi import FastAPI

from .config import settings

app = FastAPI()


@app.get("/")
def read_root():
    return {"environment": settings.ENVIRONMENT, "message": "Welcome to the RAG MVP!"}
