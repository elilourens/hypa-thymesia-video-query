import cv2
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
from tqdm import tqdm


class VideoQuerySystem:
    def __init__(self, db_path="./chroma_db"):
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="video_frames",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Ready! Using {self.device}")

    def extract_frames(self, video_path, frame_interval=1.5):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_step = int(fps * frame_interval)
        frames = []

        print(f"Extracting frames from {video_path}")
        print(f"Video: {duration:.2f}s, {fps:.2f} fps, sampling every {frame_interval}s")

        frame_count = 0
        with tqdm(total=int(duration / frame_interval)) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_step == 0:
                    timestamp = frame_count / fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((frame_rgb, timestamp))
                    pbar.update(1)
                frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames")
        return frames

    def _embed_image(self, image):
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def _embed_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def index_video(self, video_path, frame_interval=1.5, video_id=None):
        if video_id is None:
            video_id = Path(video_path).stem

        frames = self.extract_frames(video_path, frame_interval)

        print("Generating embeddings and indexing...")
        embeddings = []
        metadatas = []
        ids = []

        for idx, (frame, timestamp) in enumerate(tqdm(frames)):
            embedding = self._embed_image(frame)
            embeddings.append(embedding.tolist())
            metadatas.append({
                "video_id": video_id,
                "timestamp": timestamp,
                "frame_index": idx,
                "video_path": str(video_path)
            })
            ids.append(f"{video_id}_frame_{idx}")

        self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
        print(f"Successfully indexed {len(frames)} frames from {video_id}")

    def search(self, query, n_results=5, video_id=None):
        print(f"Searching for: '{query}'")
        query_embedding = self._embed_text(query)
        where_filter = {"video_id": video_id} if video_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
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

        return formatted_results
