import logging

import chromadb
from chromadb.api import ClientAPI

from src.config import settings

log = logging.getLogger("uvicorn")


class ChromaDBService:
    """
    Manages the connection to ChromaDB and collection operations.
    """

    _client: ClientAPI | None = None

    @classmethod
    def get_client(cls) -> ClientAPI:
        """
        Initializes and returns a singleton ChromaDB CloudClient instance.
        """
        if cls._client is None:
            log.info("Initializing ChromaDB client...")
            try:
                cls._client = chromadb.CloudClient(
                    api_key=settings.CHROMA_API_KEY,
                    tenant=settings.CHROMA_TENANT,
                    database=settings.CHROMA_DATABASE,
                )
                log.info("ChromaDB client initialized successfully.")
            except Exception as e:
                log.error(f"Failed to initialize ChromaDB client: {e}")
                raise RuntimeError("Could not connect to ChromaDB.") from e
        return cls._client

    @classmethod
    def get_or_create_collection(cls, name: str) -> chromadb.Collection:
        """
        Gets or creates a collection.

        Args:
            name: The name of the collection.

        Returns:
            A ChromaDB Collection object.
        """
        client = cls.get_client()
        try:
            log.info(f"Getting or creating collection '{name}'...")
            collection = client.get_or_create_collection(name=name)
            log.info(f"Successfully accessed collection '{name}'.")
            return collection
        except Exception as e:
            log.error(f"Failed to get or create collection '{name}': {e}")
            raise


chroma_service = ChromaDBService()
