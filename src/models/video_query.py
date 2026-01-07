from pathlib import Path
from tqdm import tqdm
from src.embeddings.clip_embedder import CLIPEmbedder
from src.storage.database import VideoDatabase
from src.video.processor import VideoProcessor


class VideoQuerySystem:
    def __init__(self, db_path="./chroma_db"):
        self.embedder = CLIPEmbedder()
        self.database = VideoDatabase(db_path)
        self.video_processor = VideoProcessor()
        print("Ready!")

    def extract_frames(self, video_path, frame_interval=1.5, skip_solid_frames=True, save_frames_to_disk=True):
        return self.video_processor.extract_frames(video_path, frame_interval, skip_solid_frames, save_frames_to_disk)

    def index_video(self, video_path, frame_interval=1.5, video_id=None, skip_solid_frames=True, save_frames_to_disk=True, detect_scenes=True, scene_threshold=30.0):
        if video_id is None:
            video_id = Path(video_path).stem

        frames = self.extract_frames(video_path, frame_interval, skip_solid_frames, save_frames_to_disk)

        # Detect scene changes
        scene_ids = None
        if detect_scenes:
            scene_ids = self.video_processor.detect_scene_changes(frames, scene_threshold)

        print("Generating embeddings and indexing...")
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

    def search(self, query, n_results=5, video_id=None, diversify=True, diversity_weight=0.5):
        print(f"Searching for: '{query}'")
        query_embedding = self.embedder.embed_text(query)
        where_filter = {"video_id": video_id} if video_id else None
        return self.database.query(query_embedding, n_results, where_filter, diversify, diversity_weight)

    def get_frame_count(self):
        return self.database.count()

    @property
    def client(self):
        return self.database.get_client()

    @property
    def collection(self):
        return self.database.get_collection()
