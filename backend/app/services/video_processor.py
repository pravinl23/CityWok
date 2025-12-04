import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Optional
import numpy as np
from app.core.config import settings

class VideoProcessor:
    def __init__(self):
        self.model_id = settings.CLIP_MODEL_ID
        
        # Detect best available hardware acceleration
        if torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 Using NVIDIA CUDA acceleration")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            print("🚀 Using Apple Metal (MPS) acceleration")
        else:
            self.device = "cpu"
            print("⚠️ Using CPU (consider GPU for faster processing)")
            
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
                
                # Warmup: Run dummy inference to load CUDA kernels (faster first real query)
                if self.device in ("cuda", "mps"):
                    dummy = Image.new('RGB', (224, 224), color='red')
                    inputs = self.processor(images=[dummy], return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        _ = self.model.get_image_features(**inputs)
                    print("✓ Model warmup complete")
                    
                self._model_loaded = True
                print("CLIP model loaded successfully.")
            except Exception as e:
                print(f"Error loading CLIP model: {e}")
                raise

    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute histogram difference between two frames for scene detection."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute histograms
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        
        # Normalize
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
        # Compare using correlation (1 = identical, -1 = opposite)
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return 1.0 - score  # Convert to difference (0 = identical)

    def extract_keyframes(self, video_path: str, 
                          max_frames: int = 150,
                          scene_threshold: float = 0.3,
                          min_interval: float = 0.5) -> List[Tuple[float, Image.Image]]:
        """
        Extract keyframes using scene change detection.
        
        Instead of brute-forcing every frame (1800/min at 30fps),
        this detects scene changes and high-motion moments to get
        50-150 distinctive frames per episode.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of keyframes to extract
            scene_threshold: Histogram difference threshold for scene change (0-1)
            min_interval: Minimum seconds between keyframes
        """
        frames = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"📹 Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
            
            prev_frame = None
            prev_timestamp = -min_interval
            frame_count = 0
            
            # Also sample at regular intervals to ensure coverage
            interval_frames = max(1, int(total_frames / (max_frames * 2)))
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame is None or frame.size == 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / fps
                
                # Check if this is a keyframe
                is_keyframe = False
                
                # 1. Scene change detection
                if prev_frame is not None:
                    diff = self._compute_frame_difference(prev_frame, frame)
                    if diff > scene_threshold and (timestamp - prev_timestamp) >= min_interval:
                        is_keyframe = True
                else:
                    # First frame is always a keyframe
                    is_keyframe = True
                
                # 2. Regular interval sampling (backup)
                if frame_count % interval_frames == 0 and (timestamp - prev_timestamp) >= min_interval:
                    is_keyframe = True
                
                if is_keyframe:
                    try:
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_frame)
                            frames.append((timestamp, pil_image))
                            prev_timestamp = timestamp
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                
                prev_frame = frame.copy()
                frame_count += 1
                
            print(f"✓ Extracted {len(frames)} keyframes (reduced from {frame_count} total frames)")
            
        except Exception as e:
            print(f"Error in extract_keyframes: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if cap is not None:
                cap.release()
                
        return frames

    def extract_frames(self, video_path: str, sampling_rate: int = 1) -> List[Tuple[float, Image.Image]]:
        """
        Legacy method: Extracts frames at fixed sampling rate.
        For new code, prefer extract_keyframes() for better efficiency.
        """
        frames = []
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            interval = int(fps / sampling_rate) if sampling_rate < fps else 1
            
            frame_count = 0
            max_frames = 1000
            max_iterations = 50000
            
            while len(frames) < max_frames and frame_count < max_iterations:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame is None or frame.size == 0:
                    frame_count += 1
                    continue
                    
                if frame_count % interval == 0:
                    try:
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

    def compute_embeddings(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Computes CLIP embeddings for a list of images with batching.
        
        Args:
            images: List of PIL images
            batch_size: Number of images to process in each GPU batch
        """
        if not images:
            return np.array([])

        self._ensure_model_loaded()
        
        all_embeddings = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            batch_embeddings = image_features.cpu().numpy().astype(np.float32)
            all_embeddings.append(batch_embeddings)
            
            if (i + batch_size) % 100 == 0:
                print(f"Embedded {min(i + batch_size, len(images))}/{len(images)} images...")
        
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Safety check
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("Warning: Found NaN or Inf in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        
        return embeddings

# Global instance
video_processor = VideoProcessor()
