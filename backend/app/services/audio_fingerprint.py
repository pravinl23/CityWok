import os
import numpy as np
import librosa
import pickle
import hashlib
from typing import List, Dict, Any, Tuple
from app.core.config import settings

class AudioFingerprinter:
    def __init__(self):
        self.db_path = os.path.join(settings.DATA_DIR, "audio_fingerprints.pkl")
        # fingerprint_hash -> [(episode_id, offset_seconds), ...]
        self.fingerprints: Dict[str, List[Tuple[str, float]]] = {}
        self.load_db()
        
    def load_db(self):
        """Load fingerprints from disk."""
        if os.path.exists(self.db_path):
            print(f"Loading audio fingerprints from {self.db_path}...")
            try:
                with open(self.db_path, 'rb') as f:
                    self.fingerprints = pickle.load(f)
                print(f"Loaded {len(self.fingerprints)} audio hashes.")
            except Exception as e:
                print(f"Error loading audio DB: {e}")
                self.fingerprints = {}
        else:
            print("No audio fingerprint database found. Starting fresh.")

    def save_db(self):
        """Save fingerprints to disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.fingerprints, f)
            print(f"Saved {len(self.fingerprints)} audio hashes to {self.db_path}")
        except Exception as e:
            print(f"Error saving audio DB: {e}")
        
    def fingerprint_audio(self, file_path: str) -> List[Tuple[str, float]]:
        """
        Generate fingerprints using Chroma N-grams.
        Robust to noise and simple pitch shifts.
        """
        try:
            # Load with librosa (resample to 22050Hz for consistency)
            y, sr = librosa.load(file_path, sr=22050, mono=True)
            
            # Compute Chroma features (pitch class profiles)
            # hop_length=512 -> ~43 frames/sec
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            
            # Get dominant note at each frame (0-11)
            notes = np.argmax(chroma, axis=0)
            
            hashes = []
            # Create N-grams
            N = 4  # 4 consecutive frames ~ 100ms
            for i in range(len(notes) - N):
                # Create a tuple of the N notes
                ngram = tuple(notes[i:i+N])
                
                # Hash the ngram
                h = hashlib.md5(str(ngram).encode()).hexdigest()
                
                # Get time of the *start* of the ngram
                time_offset = librosa.frames_to_time(i, sr=sr, hop_length=512)
                
                hashes.append((h, time_offset))
                
            return hashes
            
        except Exception as e:
            print(f"Error fingerprinting audio {file_path}: {e}")
            return []

    def add_episode(self, episode_id: str, file_path: str):
        """
        Process an episode video/audio and store its fingerprints.
        """
        print(f"Fingerprinting audio for {episode_id}...")
        fingerprints = self.fingerprint_audio(file_path)
        if not fingerprints:
            print(f"No audio fingerprints generated for {episode_id}")
            return
            
        count = 0
        for h, offset in fingerprints:
            if h not in self.fingerprints:
                self.fingerprints[h] = []
            self.fingerprints[h].append((episode_id, offset))
            count += 1
            
        print(f"Added {count} audio hashes for {episode_id}")
        self.save_db()

    def match_clip(self, file_path: str) -> Dict[str, Any]:
        """
        Match a query clip against the database.
        """
        print("Fingerprinting query clip...")
        query_prints = self.fingerprint_audio(file_path)
        if not query_prints:
            print("No fingerprints found in query.")
            return {}
            
        print(f"Generated {len(query_prints)} query hashes. Searching DB...")
        
        # Find matches
        matches: Dict[str, List[float]] = {}
        match_count = 0
        
        for h, t_clip in query_prints:
            if h in self.fingerprints:
                for ep_id, t_ep in self.fingerprints[h]:
                    diff = t_ep - t_clip
                    if ep_id not in matches:
                        matches[ep_id] = []
                    matches[ep_id].append(diff)
                    match_count += 1
        
        print(f"Found {match_count} raw hash matches across {len(matches)} episodes.")
        
        if not matches:
            return {}
            
        # Find the best episode: one with the most consistent time difference
        best_episode = None
        best_count = 0
        best_offset = 0
        
        for ep_id, diffs in matches.items():
            if len(diffs) < 10: # Minimum matching hashes
                continue
                
            # Bin the differences to find the mode (alignment)
            # Round to nearest 0.5s to allow for slight timing variations
            rounded_diffs = [round(d * 2) / 2 for d in diffs]
            
            from collections import Counter
            counts = Counter(rounded_diffs)
            
            if not counts:
                continue
                
            most_common_diff, count = counts.most_common(1)[0]
            
            if count > best_count:
                best_count = count
                best_episode = ep_id
                best_offset = most_common_diff
                
        print(f"Best audio match: {best_episode} with {best_count} aligned hashes at {best_offset}s")
        
        if best_episode and best_count > 20: # Confidence threshold
            return {
                "episode_id": best_episode,
                "timestamp": max(0, best_offset),
                "confidence": best_count,
                "method": "audio"
            }
            
        return {}

# Global instance
audio_matcher = AudioFingerprinter()

