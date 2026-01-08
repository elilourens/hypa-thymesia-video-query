import torch
from sentence_transformers import SentenceTransformer
import numpy as np


class TranscriptEmbedder:
    """Generates embeddings for text transcripts using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the transcript embedder.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading transcript embedder on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Transcript embedder loaded: {model_name}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.flatten()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of normalized embeddings
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings
