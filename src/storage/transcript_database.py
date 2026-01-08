import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import chromadb
from chromadb.config import Settings


class TranscriptDatabase:
    """Manages storage and retrieval of transcript embeddings in ChromaDB."""

    def __init__(self, db_path="./chroma_db"):
        """
        Initialize ChromaDB client for transcript storage.

        Args:
            db_path: Path to ChromaDB persistent storage directory
        """
        print("Initializing ChromaDB for transcripts...")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="video_transcripts",
            metadata={"hnsw:space": "cosine"}
        )
        print("Transcript database ready!")

    def add_transcripts(self, embeddings, metadatas, ids):
        """
        Add transcript embeddings to the database.

        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries (video_id, start_time, end_time, text)
            ids: List of unique IDs for each transcript chunk
        """
        self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, query_embedding, n_results=5, where_filter=None, diversify=True, diversity_weight=0.5):
        """
        Query transcript database for similar chunks.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter
            diversify: Whether to apply temporal diversity
            diversity_weight: Weight for diversity vs relevance (0-1)

        Returns:
            List of result dictionaries with metadata and similarity scores
        """
        # Fetch more results than needed to enable diversity selection
        fetch_multiplier = 3 if diversify else 1
        fetch_count = min(n_results * fetch_multiplier, self.collection.count())

        if fetch_count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_count,
            where=where_filter
        )

        formatted_results = []
        for idx in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][idx],
                'metadata': results['metadatas'][0][idx],
                'distance': results['distances'][0][idx],
                'similarity': 1 - results['distances'][0][idx]
            })

        # Apply temporal diversity if enabled
        if diversify and len(formatted_results) > n_results:
            formatted_results = self._diversify_results(formatted_results, n_results, diversity_weight)
        else:
            formatted_results = formatted_results[:n_results]

        return formatted_results

    def _diversify_results(self, results, n_results, diversity_weight=0.5):
        """
        Re-rank results to maximize temporal diversity while maintaining relevance.

        Args:
            results: List of result dictionaries with metadata
            n_results: Number of results to return
            diversity_weight: Weight for diversity vs relevance (0-1)
        """
        if len(results) <= n_results:
            return results

        selected = []
        remaining = results.copy()

        # Always select the best match first
        selected.append(remaining.pop(0))

        # Greedily select remaining results
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                # Calculate temporal diversity
                min_time_diff = float('inf')
                candidate_start = float(candidate['metadata']['start_time'])

                for selected_result in selected:
                    selected_start = float(selected_result['metadata']['start_time'])
                    time_diff = abs(candidate_start - selected_start)
                    min_time_diff = min(min_time_diff, time_diff)

                # Normalize temporal diversity (20s = full diversity score)
                diversity_score = min(min_time_diff / 20.0, 1.0)

                # Relevance score
                relevance_score = candidate['similarity']

                # Combined score
                combined_score = (1 - diversity_weight) * relevance_score + diversity_weight * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected

    def clear_collection(self):
        """Clear all transcripts from the database."""
        self.client.delete_collection(name="video_transcripts")
        self.collection = self.client.get_or_create_collection(
            name="video_transcripts",
            metadata={"hnsw:space": "cosine"}
        )

    def count(self):
        """Get total number of transcript chunks in the database."""
        return self.collection.count()

    def get_client(self):
        """Get the ChromaDB client."""
        return self.client

    def get_collection(self):
        """Get the transcript collection."""
        return self.collection
