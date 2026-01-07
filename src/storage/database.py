import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import chromadb
from chromadb.config import Settings


class VideoDatabase:
    def __init__(self, db_path="./chroma_db"):
        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="video_frames",
            metadata={"hnsw:space": "cosine"}
        )
        print("Database ready!")

    def add_frames(self, embeddings, metadatas, ids):
        self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, query_embedding, n_results=5, where_filter=None, diversify=True, diversity_weight=0.5):
        # Fetch more results than needed to enable diversity selection
        fetch_multiplier = 3 if diversify else 1
        fetch_count = min(n_results * fetch_multiplier, self.collection.count())

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
        Re-rank results to maximize scene diversity while maintaining relevance.

        Uses a greedy selection algorithm that balances:
        - Semantic similarity (how well it matches the query)
        - Scene diversity (preferring results from different scenes)

        Args:
            results: List of result dictionaries with metadata
            n_results: Number of results to return
            diversity_weight: Weight for diversity vs relevance (0-1).
                            0 = pure relevance, 1 = pure diversity, 0.5 = balanced
        """
        if len(results) <= n_results:
            return results

        selected = []
        remaining = results.copy()
        selected_scenes = set()

        # Always select the best match first
        first_result = remaining.pop(0)
        selected.append(first_result)
        if 'scene_id' in first_result['metadata']:
            selected_scenes.add(first_result['metadata']['scene_id'])

        # Check if we have scene_id data
        has_scene_ids = 'scene_id' in first_result['metadata']

        # Greedily select remaining results
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                if has_scene_ids:
                    # Scene-based diversity
                    candidate_scene = candidate['metadata'].get('scene_id')

                    # Diversity score: 1.0 if from new scene, 0.0 if from existing scene
                    diversity_score = 1.0 if candidate_scene not in selected_scenes else 0.0
                else:
                    # Fallback to time-based diversity if no scene IDs
                    min_time_diff = float('inf')
                    candidate_time = candidate['metadata']['timestamp']

                    for selected_result in selected:
                        selected_time = selected_result['metadata']['timestamp']
                        time_diff = abs(candidate_time - selected_time)
                        min_time_diff = min(min_time_diff, time_diff)

                    # Normalize temporal diversity (10s = full diversity score)
                    diversity_score = min(min_time_diff / 10.0, 1.0)

                # Relevance score
                relevance_score = candidate['similarity']

                # Combined score
                combined_score = (1 - diversity_weight) * relevance_score + diversity_weight * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx

            selected_result = remaining.pop(best_idx)
            selected.append(selected_result)

            # Track selected scene
            if has_scene_ids and 'scene_id' in selected_result['metadata']:
                selected_scenes.add(selected_result['metadata']['scene_id'])

        return selected

    def clear_collection(self):
        self.client.delete_collection(name="video_frames")
        self.collection = self.client.get_or_create_collection(
            name="video_frames",
            metadata={"hnsw:space": "cosine"}
        )

    def count(self):
        return self.collection.count()

    def get_client(self):
        return self.client

    def get_collection(self):
        return self.collection
