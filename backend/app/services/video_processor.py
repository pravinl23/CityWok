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
        self.processor = None
        self.model = None
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Lazy load the model only when needed."""
        if not self._model_loaded:
            print(f"Loading CLIP model {self.model_id} on {self.device}...")
            try:
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
                self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
                self.model.eval()
                self._model_loaded = True
                print("CLIP model loaded successfully.")
            except Exception as e:
                print(f"Error loading CLIP model: {e}")
                raise

    def extract_frames(self, video_path: str, sampling_rate: int = 1) -> List[Tuple[float, Image.Image]]:
        """
        Extracts frames from a video at the given sampling rate.
        Returns a list of (timestamp, PIL Image) tuples.
        """
        frames = []
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback
            interval = int(fps / sampling_rate) if sampling_rate < fps else 1
            
            frame_count = 0
            max_frames = 1000  # Allow enough frames for full episode processing (~700-800 for 22min episodes at 0.5fps)
            max_iterations = 50000  # Safety limit to prevent infinite loops
            
            while len(frames) < max_frames and frame_count < max_iterations:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip empty frames
                if frame is None or frame.size == 0:
                    frame_count += 1
                    continue
                    
                if frame_count % interval == 0:
                    try:
                        # Convert BGR (OpenCV) to RGB (PIL)
                        if len(frame.shape) != 3 or frame.shape[2] != 3:
                            frame_count += 1
                            continue
                            
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        if timestamp < 0:
                            timestamp = frame_count / fps
                        frames.append((timestamp, pil_image))
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                        frame_count += 1
                        continue
                
                frame_count += 1
        except Exception as e:
            print(f"Error in extract_frames: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if cap is not None:
                cap.release()
        return frames

    def compute_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """
        Computes CLIP embeddings for a list of images.
        Returns a numpy array of embeddings.
        """
        if not images:
            return np.array([])

        self._ensure_model_loaded()
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize embeddings
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        # Convert to numpy and ensure float32
        embeddings = image_features.cpu().numpy().astype(np.float32)
        
        # Safety check: ensure no NaN or Inf values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("Warning: Found NaN or Inf in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            # Renormalize after cleaning
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        
        return embeddings

# Global instance
video_processor = VideoProcessor()

