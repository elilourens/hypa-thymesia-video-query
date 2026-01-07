import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"CLIP model loaded on {self.device}")

    def embed_image(self, image):
        pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def embed_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()
