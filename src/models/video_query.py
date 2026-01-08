from pathlib import Path
from tqdm import tqdm
from src.embeddings.clip_embedder import CLIPEmbedder
from src.embeddings.transcript_embedder import TranscriptEmbedder
from src.storage.database import VideoDatabase
from src.storage.transcript_database import TranscriptDatabase
from src.video.processor import VideoProcessor
from src.audio.audio_processor import AudioProcessor


class VideoQuerySystem:
    def __init__(self, db_path="./chroma_db", whisper_model="base"):
        self.embedder = CLIPEmbedder()
        self.transcript_embedder = TranscriptEmbedder()
        self.database = VideoDatabase(db_path)
        self.transcript_database = TranscriptDatabase(db_path)
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor(whisper_model)
        print("Ready!")

    def extract_frames(self, video_path, frame_interval=1.5, skip_solid_frames=True, save_frames_to_disk=True):
        return self.video_processor.extract_frames(video_path, frame_interval, skip_solid_frames, save_frames_to_disk)

    def index_video(self, video_path, frame_interval=1.5, video_id=None, skip_solid_frames=True, save_frames_to_disk=True, detect_scenes=True, scene_threshold=30.0, transcribe=True, chunk_duration=20.0):
        if video_id is None:
            video_id = Path(video_path).stem

        frames = self.extract_frames(video_path, frame_interval, skip_solid_frames, save_frames_to_disk)

        # Detect scene changes
        scene_ids = None
        if detect_scenes:
            scene_ids = self.video_processor.detect_scene_changes(frames, scene_threshold)

        print("Generating frame embeddings and indexing...")
        embeddings = []
        metadatas = []
        ids = []

        for idx, (frame, timestamp) in enumerate(tqdm(frames)):
            embedding = self.embedder.embed_image(frame)
            embeddings.append(embedding.tolist())

            metadata = {
                "video_id": video_id,
                "timestamp": timestamp,
                "frame_index": idx,
                "video_path": str(video_path)
            }

            # Add scene ID if scene detection was performed
            if scene_ids is not None:
                metadata["scene_id"] = scene_ids[idx]

            metadatas.append(metadata)
            ids.append(f"{video_id}_frame_{idx}")

        self.database.add_frames(embeddings, metadatas, ids)
        print(f"Successfully indexed {len(frames)} frames from {video_id}")

        # Transcribe and index audio
        if transcribe:
            self.index_transcript(video_path, video_id, chunk_duration)

    def index_transcript(self, video_path, video_id=None, chunk_duration=20.0):
        """
        Transcribe video audio and index transcript chunks.

        Args:
            video_path: Path to video file
            video_id: Optional video identifier
            chunk_duration: Duration of each transcript chunk in seconds
        """
        if video_id is None:
            video_id = Path(video_path).stem

        # Transcribe audio
        transcript_data = self.audio_processor.transcribe(video_path)

        # Chunk transcript
        chunks = self.audio_processor.chunk_transcript(transcript_data, chunk_duration)

        if not chunks:
            print("No transcript chunks generated.")
            return

        print(f"Generating embeddings for {len(chunks)} transcript chunks...")

        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.transcript_embedder.embed_batch(texts)

        # Prepare data for database
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            metadata = {
                "video_id": video_id,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "text": chunk["text"],
                "video_path": str(video_path)
            }
            metadatas.append(metadata)
            ids.append(f"{video_id}_transcript_{idx}")

        # Add to database
        self.transcript_database.add_transcripts(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        print(f"Successfully indexed {len(chunks)} transcript chunks from {video_id}")

    def search(self, query, n_results=5, video_id=None, diversify=True, diversity_weight=0.5):
        """Search for frames matching the query."""
        print(f"Searching frames for: '{query}'")
        query_embedding = self.embedder.embed_text(query)
        where_filter = {"video_id": video_id} if video_id else None
        return self.database.query(query_embedding, n_results, where_filter, diversify, diversity_weight)

    def search_transcripts(self, query, n_results=5, video_id=None, diversify=True, diversity_weight=0.5):
        """Search for transcript chunks matching the query."""
        print(f"Searching transcripts for: '{query}'")
        query_embedding = self.transcript_embedder.embed_text(query)
        where_filter = {"video_id": video_id} if video_id else None
        return self.transcript_database.query(query_embedding, n_results, where_filter, diversify, diversity_weight)

    def search_combined(self, query, n_results=5, video_id=None, diversify=True, diversity_weight=0.5):
        """
        Search both frames and transcripts, returning combined results.

        Args:
            query: Search query
            n_results: Total number of results to return
            video_id: Optional filter by video
            diversify: Whether to apply diversity
            diversity_weight: Diversity vs relevance weight

        Returns:
            Dictionary with 'frames' and 'transcripts' keys containing respective results
        """
        print(f"Searching frames and transcripts for: '{query}'")

        # Search both collections
        frame_results = self.search(query, n_results, video_id, diversify, diversity_weight)
        transcript_results = self.search_transcripts(query, n_results, video_id, diversify, diversity_weight)

        return {
            "frames": frame_results,
            "transcripts": transcript_results
        }

    def get_frame_count(self):
        """Get number of indexed frames."""
        return self.database.count()

    def get_transcript_count(self):
        """Get number of indexed transcript chunks."""
        return self.transcript_database.count()

    def clear_all(self):
        """Clear both frame and transcript databases."""
        self.database.clear_collection()
        self.transcript_database.clear_collection()

    @property
    def client(self):
        return self.database.get_client()

    @property
    def collection(self):
        return self.database.get_collection()
