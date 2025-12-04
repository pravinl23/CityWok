"""
Audio Fingerprinting Service

Implements Shazam-style spectral peak landmark fingerprinting:
1. Convert audio to spectrogram
2. Find local maxima (peaks) in time-frequency space
3. Create "constellation map" of peaks
4. Hash pairs of peaks with their relative timing
5. Store in inverted index for fast lookup

This is much more robust than simple chroma N-grams because:
- Peaks survive noise, compression, and volume changes
- Time-aligned pairs eliminate false positives
- O(1) hash lookup instead of O(N) comparison
"""

import os
import numpy as np
import librosa
import pickle
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from scipy.ndimage import maximum_filter
from app.core.config import settings


class AudioFingerprinter:
    """Shazam-style audio fingerprinting with spectral peak landmarks."""
    
    def __init__(self):
        self.db_path = os.path.join(settings.DATA_DIR, "audio_fingerprints.pkl")
        
        # Fingerprint parameters (tuned for TV audio)
        self.sr = 22050                 # Sample rate
        self.n_fft = 2048               # FFT window size
        self.hop_length = 512           # ~23ms per frame
        self.peak_neighborhood = 20     # Local max filter size
        self.fan_value = 15             # Number of peaks to pair with each anchor
        self.min_time_delta = 0         # Min time between paired peaks (frames)
        self.max_time_delta = 200       # Max time between paired peaks (frames)
        self.min_freq = 300             # Min frequency (Hz) - skip low rumble
        self.max_freq = 8000            # Max frequency (Hz) - skip high noise
        
        # Database: hash -> [(episode_id, anchor_time), ...]
        self.fingerprints: Dict[str, List[Tuple[str, float]]] = {}
        self.episode_hash_counts: Dict[str, int] = {}  # Track hashes per episode
        
        self.load_db()
        
    def load_db(self):
        """Load fingerprints from disk."""
        if os.path.exists(self.db_path):
            print(f"Loading audio fingerprints from {self.db_path}...")
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'fingerprints' in data:
                        self.fingerprints = data['fingerprints']
                        self.episode_hash_counts = data.get('counts', {})
                    else:
                        # Legacy format
                        self.fingerprints = data
                        self.episode_hash_counts = {}
                print(f"✓ Loaded {len(self.fingerprints)} unique audio hashes")
            except Exception as e:
                print(f"Error loading audio DB: {e}")
                self.fingerprints = {}
                self.episode_hash_counts = {}
        else:
            print("No audio fingerprint database found. Starting fresh.")

    def clear_db(self):
        """Clear all fingerprints from memory and disk."""
        self.fingerprints = {}
        self.episode_hash_counts = {}
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        print("✓ Audio database cleared")

    def save_db(self):
        """Save fingerprints to disk (atomic write to prevent corruption)."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            # Write to temp file first, then rename atomically
            temp_path = self.db_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    'fingerprints': self.fingerprints,
                    'counts': self.episode_hash_counts
                }, f)
            # Atomic rename (prevents corruption if interrupted)
            os.replace(temp_path, self.db_path)
            print(f"✓ Saved {len(self.fingerprints)} audio hashes")
        except Exception as e:
            print(f"Error saving audio DB: {e}")
            # Clean up temp file if it exists
            temp_path = self.db_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def _get_spectrogram_peaks(self, y: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find local maxima in spectrogram (constellation map).
        Returns list of (time_frame, freq_bin) tuples.
        """
        # Compute spectrogram
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        
        # Convert to dB scale
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        # Find local maxima
        local_max = maximum_filter(S_db, size=self.peak_neighborhood) == S_db
        
        # Apply threshold (only keep peaks above -60dB)
        threshold = S_db.max() - 60
        local_max &= (S_db > threshold)
        
        # Convert frequency limits to bins
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        min_bin = np.searchsorted(freq_bins, self.min_freq)
        max_bin = np.searchsorted(freq_bins, self.max_freq)
        
        # Mask out frequencies outside range
        local_max[:min_bin, :] = False
        local_max[max_bin:, :] = False
        
        # Get peak coordinates
        peaks = np.argwhere(local_max)  # (freq_bin, time_frame)
        
        # Convert to (time, freq) format and sort by time
        peaks = [(int(p[1]), int(p[0])) for p in peaks]
        peaks.sort(key=lambda x: x[0])
        
        return peaks

    def _create_hashes(self, peaks: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        """
        Create fingerprint hashes from peak pairs.
        
        Each hash encodes:
        - Anchor frequency
        - Target frequency  
        - Time delta between them
        
        Returns list of (hash, anchor_time_seconds) tuples.
        """
        hashes = []
        
        for i, (t1, f1) in enumerate(peaks):
            # Pair with nearby peaks (fan out)
            for j in range(i + 1, min(i + self.fan_value + 1, len(peaks))):
                t2, f2 = peaks[j]
                dt = t2 - t1
                
                if self.min_time_delta <= dt <= self.max_time_delta:
                    # Create hash from freq1, freq2, time_delta
                    # Quantize frequencies to 64 bins for robustness
                    f1_q = f1 // 8
                    f2_q = f2 // 8
                    dt_q = dt // 4  # Quantize time delta too
                    
                    hash_input = f"{f1_q}|{f2_q}|{dt_q}"
                    h = hashlib.md5(hash_input.encode()).hexdigest()[:16]
                    
                    # Store anchor time in seconds
                    anchor_time = t1 * self.hop_length / self.sr
                    hashes.append((h, anchor_time))
        
        return hashes

    def fingerprint_audio(self, file_path: str) -> List[Tuple[str, float]]:
        """
        Generate fingerprints for an audio/video file.
        Returns list of (hash, time_offset) tuples.
        """
        try:
            print(f"🎵 Fingerprinting: {os.path.basename(file_path)}")
            
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            duration = len(y) / sr
            print(f"   Duration: {duration:.1f}s")
            
            # Find spectral peaks
            peaks = self._get_spectrogram_peaks(y)
            print(f"   Found {len(peaks)} spectral peaks")
            
            # Create hashes from peak pairs
            hashes = self._create_hashes(peaks)
            print(f"   Generated {len(hashes)} fingerprint hashes")
            
            return hashes
            
        except Exception as e:
            print(f"Error fingerprinting audio {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def add_episode(self, episode_id: str, file_path: str):
        """Process an episode and store its fingerprints."""
        fingerprints = self.fingerprint_audio(file_path)
        
        if not fingerprints:
            print(f"⚠️ No audio fingerprints generated for {episode_id}")
            return
        
        # Add to database
        count = 0
        for h, offset in fingerprints:
            if h not in self.fingerprints:
                self.fingerprints[h] = []
            self.fingerprints[h].append((episode_id, offset))
            count += 1
        
        self.episode_hash_counts[episode_id] = count
        print(f"✓ Added {count} audio hashes for {episode_id}")
        self.save_db()

    def match_clip(self, file_path: str) -> Dict[str, Any]:
        """
        Match a query clip against the database.
        
        Uses time-aligned voting: only counts matches where
        multiple hashes agree on the same time offset.
        This eliminates false positives from coincidental matches.
        
        OPTIMIZED: Skips common hashes and uses early termination.
        """
        print("🔍 Fingerprinting query clip...")
        query_prints = self.fingerprint_audio(file_path)
        
        if not query_prints:
            print("No fingerprints found in query.")
            return {}
        
        if not self.fingerprints:
            print("Audio database is empty.")
            return {}
        
        # OPTIMIZATION 1: Filter out overly common hashes (appear in >10 episodes)
        # These are likely noise and slow down matching
        max_episodes_per_hash = 10
        filtered_prints = []
        skipped_common = 0
        for h, t_query in query_prints:
            if h in self.fingerprints:
                # Skip hashes that appear in too many episodes (likely common sounds)
                if len(self.fingerprints[h]) > max_episodes_per_hash:
                    skipped_common += 1
                    continue
                filtered_prints.append((h, t_query))
        
        if skipped_common > 0:
            print(f"⏭️  Skipped {skipped_common} overly common hashes (appear in >{max_episodes_per_hash} episodes)")
        
        # OPTIMIZATION 2: Sample query hashes if too many (use every Nth hash)
        # This speeds up long clips without losing accuracy
        max_query_hashes = 10000
        if len(filtered_prints) > max_query_hashes:
            step = len(filtered_prints) // max_query_hashes
            filtered_prints = filtered_prints[::step]
            print(f"📊 Sampling: Using {len(filtered_prints)} of {len(query_prints)} query hashes")
        
        print(f"Searching {len(filtered_prints)} query hashes against {len(self.fingerprints)} DB hashes...")
        
        # Find matches: episode -> list of (db_time - query_time) offsets
        matches: Dict[str, List[float]] = defaultdict(list)
        match_count = 0
        
        for h, t_query in filtered_prints:
            if h in self.fingerprints:
                for ep_id, t_db in self.fingerprints[h]:
                    offset = t_db - t_query
                    matches[ep_id].append(offset)
                    match_count += 1
        
        print(f"Found {match_count} raw hash matches across {len(matches)} episodes")
        
        if not matches:
            return {}
        
        # Find best match using time-aligned voting
        # OPTIMIZATION 3: Early termination - stop if we find a clear winner
        best_episode = None
        best_count = 0
        best_offset = 0.0
        
        # Sort episodes by match count (most likely first)
        sorted_episodes = sorted(matches.items(), key=lambda x: len(x[1]), reverse=True)
        
        for ep_id, offsets in sorted_episodes:
            if len(offsets) < 5:  # Need minimum matches
                continue
            
            # OPTIMIZATION 4: Limit offset processing for speed
            # Only process first 5000 offsets per episode (enough for alignment)
            if len(offsets) > 5000:
                offsets = offsets[:5000]
            
            # Bin offsets to find consistent alignment (0.5s bins)
            binned = [round(o * 2) / 2 for o in offsets]
            counts = Counter(binned)
            
            if not counts:
                continue
            
            mode_offset, count = counts.most_common(1)[0]
            
            # Calculate confidence as percentage of query hashes matched
            # with consistent timing
            confidence = (count / len(filtered_prints)) * 100
            
            if count > best_count:
                best_count = count
                best_episode = ep_id
                best_offset = mode_offset
            
            # OPTIMIZATION 5: Early termination if we have a clear winner
            # If best match is 3x better than next best, we're done
            if best_count > 100 and len(sorted_episodes) > 1:
                # Check if next episode could beat us
                next_ep_id, next_offsets = sorted_episodes[1]
                if len(next_offsets) < best_count / 3:
                    print(f"✓ Early termination: Clear winner found ({best_count} vs max {len(next_offsets)})")
                    break
        
        print(f"Best match: {best_episode} with {best_count} aligned hashes at offset {best_offset:.1f}s")
        
        # Require minimum aligned matches to avoid false positives
        min_aligned = max(15, len(filtered_prints) * 0.05)  # At least 5% or 15 matches
        
        if best_episode and best_count >= min_aligned:
            confidence = min(99, int((best_count / len(filtered_prints)) * 100))
            return {
                "episode_id": best_episode,
                "timestamp": max(0, best_offset),
                "confidence": confidence,
                "aligned_matches": best_count,
                "total_matches": match_count,
                "method": "audio"
            }
        
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        total_entries = sum(len(v) for v in self.fingerprints.values())
        return {
            "unique_hashes": len(self.fingerprints),
            "total_entries": total_entries,
            "episodes": len(self.episode_hash_counts),
            "avg_hashes_per_episode": total_entries / max(1, len(self.episode_hash_counts))
        }


# Global instance
audio_matcher = AudioFingerprinter()
