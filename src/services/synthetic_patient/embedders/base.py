from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    All embedding implementations must inherit from this class and implement the required methods.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for the provided text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors produced by this provider.

        Returns:
            Integer representing the embedding dimension (e.g., 1536, 768)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name/identifier of this embedding provider.

        Returns:
            String identifier (e.g., "text-embedding-3-small", "MedEmbed-base-v0.1")
        """
        pass

