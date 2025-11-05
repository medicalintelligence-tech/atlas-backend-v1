from openai import OpenAI
from .base import EmbeddingProvider


class OpenAIEmbedder(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding models.
    
    Supports models like:
    - text-embedding-3-small (1536 dimensions, faster, cheaper)
    - text-embedding-3-large (3072 dimensions, higher quality)
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # Set dimension based on model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for the provided text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            Integer representing the embedding dimension
        """
        return self._dimensions.get(self.model, 1536)

    def get_name(self) -> str:
        """
        Get the name/identifier of this embedding provider.

        Returns:
            String identifier with model name
        """
        return self.model

