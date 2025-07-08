import logging

import openai
from openai import OpenAI

from src.config import settings

log = logging.getLogger("uvicorn")


class OpenAIService:
    """
    A reusable service to interact with the OpenAI API.

    Initializes the OpenAI client once and provides methods for various
    OpenAI functionalities like creating embeddings and chat completions.
    """

    _client: OpenAI | None = None

    @classmethod
    def get_client(cls) -> OpenAI:
        """
        Returns the OpenAI client, initializing it if necessary.
        This ensures a single instance of the client is used throughout the app.
        """
        if cls._client is None:
            log.info("Initializing OpenAI client...")
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in the environment.")

            cls._client = OpenAI(api_key=settings.OPENAI_API_KEY)
            log.info("OpenAI client initialized successfully.")
        return cls._client

    @classmethod
    def create_embedding(cls, text: str, model: str = "text-embedding-3-small") -> list[float]:
        """
        Creates an embedding for the given text using the specified model.

        Args:
            text: The input text to embed.
            model: The embedding model to use.

        Returns:
            The embedding vector as a list of floats.
        """
        try:
            client = cls.get_client()
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except openai.APIError as e:
            log.error(f"OpenAI API Error during embedding: {e}")
            raise ValueError("Failed to create embedding due to an OpenAI API error.") from e
        except Exception as e:
            log.error(f"An unexpected error occurred during embedding: {e}")
            raise ValueError("An unexpected error occurred during embedding.") from e

    @classmethod
    def create_embeddings_in_batch(cls, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        """
        Creates embeddings for a list of texts in a single batch request.

        Args:
            texts: The list of input texts to embed.
            model: The embedding model to use.

        Returns:
            A list of embedding vectors, corresponding to the order of the input texts.
        """
        try:
            client = cls.get_client()
            # Replace newlines for better performance, as recommended by OpenAI
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            response = client.embeddings.create(input=cleaned_texts, model=model)
            return [data.embedding for data in response.data]
        except openai.APIError as e:
            log.error(f"OpenAI API Error during batch embedding: {e}")
            raise ValueError("Failed to create batch embeddings due to an OpenAI API error.") from e
        except Exception as e:
            log.error(f"An unexpected error occurred during batch embedding: {e}")
            raise ValueError("An unexpected error occurred during batch embedding.") from e


openai_service = OpenAIService()
