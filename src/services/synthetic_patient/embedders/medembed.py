from sentence_transformers import SentenceTransformer
from .base import EmbeddingProvider


class MedEmbedder(EmbeddingProvider):
    """
    MedEmbed embedding provider using Hugging Face sentence-transformers.
    
    Supports medical-domain specific models like:
    - abhinand/MedEmbed-small-v0.1
    - abhinand/MedEmbed-base-v0.1 (recommended)
    - abhinand/MedEmbed-large-v0.1
    
    These models are optimized for medical text and provide better
    semantic understanding of clinical concepts compared to general embeddings.
    """

    def __init__(self, model: str = "abhinand/MedEmbed-base-v0.1"):
        """
        Initialize MedEmbed embedder.

        Args:
            model: HuggingFace model name (default: abhinand/MedEmbed-base-v0.1)
        """
        self.model_name = model
        self.model = SentenceTransformer(model)
        
        # Get actual embedding dimension from the model
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for the provided text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Encode returns numpy array, convert to list
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            Integer representing the embedding dimension
        """
        return self._dimension

    def get_name(self) -> str:
        """
        Get the name/identifier of this embedding provider.

        Returns:
            String identifier with model name
        """
        return self.model_name

