import os
import numpy as np
from scipy.signal import spectrogram
from typing import List, Dict, Any, Tuple
from app.core.config import settings
import hashlib

# Try to import pydub, but make it optional for Python 3.13 compatibility
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Audio fingerprinting will be limited.")

class AudioFingerprinter:
    def __init__(self):
        # In-memory storage for MVP
        # fingerprint_hash -> [(episode_id, offset_seconds), ...]
        self.fingerprints: Dict[str, List[Tuple[str, float]]] = {}
        
    def fingerprint_audio(self, file_path: str) -> List[Tuple[str, float]]:
        """
        Generate fingerprints for an audio file.
        Returns a list of (hash, time_offset) tuples.
        """
        if not PYDUB_AVAILABLE:
            # Fallback: use librosa or ffmpeg directly
            try:
                import librosa
                samples, sr = librosa.load(file_path, sr=11025, mono=True)
                samples = (samples * 32767).astype(np.int16)  # Convert to int16 like pydub
            except ImportError:
                print("Warning: Neither pydub nor librosa available. Audio fingerprinting disabled.")
                return []
        
        try:
            if PYDUB_AVAILABLE:
                # Load audio using pydub
                audio = AudioSegment.from_file(file_path)
                # Convert to mono, 11kHz
                audio = audio.set_channels(1).set_frame_rate(11025)
                samples = np.array(audio.get_array_of_samples())
            else:
                # Already loaded with librosa above
                pass
            
            # Generate spectrogram
            f, t, Sxx = spectrogram(samples, fs=11025, nperseg=1024, noverlap=512)
            
            # Simple peak finding: find max frequency in each time bin
            # This is a very naive simplification of Shazam
            peaks = np.argmax(Sxx, axis=0)
            
            hashes = []
            # Create hashes from localized peaks (e.g., pairs of peaks)
            # Here we just hash the peak frequency at each time step
            for i, peak_freq_idx in enumerate(peaks):
                time_offset = t[i]
                # Hash the frequency index and the time relative to a window
                # To make it robust, we'd use pairs, but for MVP:
                h = hashlib.md5(f"{peak_freq_idx}".encode()).hexdigest()
                hashes.append((h, time_offset))
                
            return hashes
        except Exception as e:
            print(f"Error processing audio {file_path}: {e}")
            return []

    def add_episode(self, episode_id: str, file_path: str):
        """
        Process an episode video/audio and store its fingerprints.
        """
        fingerprints = self.fingerprint_audio(file_path)
        for h, offset in fingerprints:
            if h not in self.fingerprints:
                self.fingerprints[h] = []
            self.fingerprints[h].append((episode_id, offset))

    def match_clip(self, file_path: str) -> Dict[str, Any]:
        """
        Match a query clip against the database.
        """
        query_prints = self.fingerprint_audio(file_path)
        if not query_prints:
            return {}
            
        # Find matches
        # episode_id -> list of time_diffs
        # If clip matches episode at offset T_ep and clip offset is T_clip,
        # then T_ep - T_clip should be constant.
        matches: Dict[str, List[float]] = {}
        
        for h, t_clip in query_prints:
            if h in self.fingerprints:
                for ep_id, t_ep in self.fingerprints[h]:
                    if ep_id not in matches:
                        matches[ep_id] = []
                    matches[ep_id].append(t_ep - t_clip)
        
        # Find the best episode: one with the most consistent time difference
        best_episode = None
        best_count = 0
        best_offset = 0
        
        for ep_id, diffs in matches.items():
            # Bin the differences to find the mode
            # Round to nearest 0.1s
            rounded_diffs = [round(d, 1) for d in diffs]
            if not rounded_diffs:
                continue
                
            from collections import Counter
            counts = Counter(rounded_diffs)
            most_common_diff, count = counts.most_common(1)[0]
            
            if count > best_count:
                best_count = count
                best_episode = ep_id
                best_offset = most_common_diff
                
        if best_episode and best_count > 5: # Threshold
            return {
                "episode_id": best_episode,
                "timestamp": best_offset,
                "confidence": best_count  # crude confidence
            }
            
        return {}

# Global instance
audio_matcher = AudioFingerprinter()

