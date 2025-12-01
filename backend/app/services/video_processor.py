import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import numpy as np
from app.core.config import settings

class VideoProcessor:
    def __init__(self):
        self.model_id = settings.CLIP_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model {self.model_id} on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def extract_frames(self, video_path: str, sampling_rate: int = 1) -> List[Tuple[float, Image.Image]]:
        """
        Extracts frames from a video at the given sampling rate.
        Returns a list of (timestamp, PIL Image) tuples.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / sampling_rate) if sampling_rate < fps else 1
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frames.append((timestamp, pil_image))
            
            frame_count += 1
            
        cap.release()
        return frames

    def compute_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """
        Computes CLIP embeddings for a list of images.
        Returns a numpy array of embeddings.
        """
        if not images:
            return np.array([])

        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize embeddings
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()

# Global instance
video_processor = VideoProcessor()

