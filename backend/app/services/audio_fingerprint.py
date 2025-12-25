import os
import re
import numpy as np
import librosa
import pickle
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from scipy.ndimage import maximum_filter
from app.core.config import settings

class AudioFingerprinter:
    """
    Optimized Pickle-based audio fingerprinting.
    Loads all data into memory using efficient Numpy arrays to fit in 8GB RAM.
    """
    
    DB_CONFIG = {
        season: f"audio_fingerprints_s{season:02d}.pkl"
        for season in range(1, 21)
    }
    
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.max_seasons = int(os.getenv('MAX_SEASONS', '20'))
        
        # Fingerprint parameters
        self.sr = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.peak_neighborhood = 20
        self.fan_value = 15
        self.min_time_delta = 0
        self.max_time_delta = 200
        self.min_freq = 300
        self.max_freq = 8000
        
        # Memory-efficient storage
        # hash (str) -> numpy array of dtype=[('ep_idx', 'u2'), ('offset', 'f4')]
        self.fingerprints: Dict[str, np.ndarray] = {}
        self.episode_list: List[str] = []
        self.episode_to_idx: Dict[str, int] = {}
        
        self._scan_databases()
        print(f"📦 Eager loading all {len(self.existing_dbs)} databases into memory (optimized)...")
        self._load_all_databases()

    def _scan_databases(self):
        self.existing_dbs = []
        for season in range(1, min(self.max_seasons + 1, 21)):
            db_file = self.DB_CONFIG.get(season)
            if db_file:
                db_path = os.path.join(self.data_dir, db_file)
                if os.path.exists(db_path):
                    self.existing_dbs.append(db_file)

    def _load_all_databases(self):
        start_time = time.time()
        dtype = [('ep_idx', 'u2'), ('offset', 'f4')]
        
        for db_file in self.existing_dbs:
            db_path = os.path.join(self.data_dir, db_file)
            try:
                print(f"📂 Loading {db_file}...")
                with open(db_path, 'rb') as f:
                    data = pickle.load(f)
                
                raw_prints = data['fingerprints'] if isinstance(data, dict) and 'fingerprints' in data else data
                
                for h, entries in raw_prints.items():
                    # Convert entries for this hash to a small numpy array
                    new_entries = []
                    for ep_id, offset in entries:
                        if ep_id not in self.episode_to_idx:
                            self.episode_to_idx[ep_id] = len(self.episode_list)
                            self.episode_list.append(ep_id)
                        new_entries.append((self.episode_to_idx[ep_id], offset))
                    
                    new_arr = np.array(new_entries, dtype=dtype)
                    
                    # Merge with existing array for this hash
                    if h in self.fingerprints:
                        self.fingerprints[h] = np.concatenate([self.fingerprints[h], new_arr])
                    else:
                        self.fingerprints[h] = new_arr
                
                print(f"   ✓ Loaded {db_file} (RAM: {sum(a.nbytes for a in self.fingerprints.values()) / 1024 / 1024:.1f}MB)")
                del data # Explicitly free memory
            except Exception as e:
                print(f"   ❌ Error loading {db_file}: {e}")

        elapsed = time.time() - start_time
        print(f"✅ Loaded {len(self.fingerprints):,} unique hashes in {elapsed:.1f}s")
        print(f"📊 Final Data Size: {sum(a.nbytes for a in self.fingerprints.values()) / 1024 / 1024:.1f}MB")

    def _get_spectrogram_peaks(self, y: np.ndarray) -> List[Tuple[int, int]]:
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        local_max = maximum_filter(S_db, size=self.peak_neighborhood) == S_db
        threshold = S_db.max() - 60
        local_max &= (S_db > threshold)
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        min_bin = np.searchsorted(freq_bins, self.min_freq)
        max_bin = np.searchsorted(freq_bins, self.max_freq)
        local_max[:min_bin, :] = False
        local_max[max_bin:, :] = False
        peaks = np.argwhere(local_max)
        peaks = [(int(p[1]), int(p[0])) for p in peaks]
        peaks.sort(key=lambda x: x[0])
        return peaks

    def _create_hashes(self, peaks: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        hashes = []
        for i, (t1, f1) in enumerate(peaks):
            for j in range(i + 1, min(i + self.fan_value + 1, len(peaks))):
                t2, f2 = peaks[j]
                dt = t2 - t1
                if self.min_time_delta <= dt <= self.max_time_delta:
                    f1_q, f2_q, dt_q = f1 // 8, f2 // 8, dt // 4
                    hash_input = f"{f1_q}|{f2_q}|{dt_q}"
                    h = hashlib.md5(hash_input.encode()).hexdigest()[:16]
                    anchor_time = t1 * self.hop_length / self.sr
                    hashes.append((h, anchor_time))
        return hashes

    def match_audio_array(self, audio_array: np.ndarray, sr: int) -> Dict[str, Any]:
        start_time = time.time()
        peaks = self._get_spectrogram_peaks(audio_array)
        query_prints = self._create_hashes(peaks)
        if not query_prints: return {}
        
        # Uniformly sample hashes from the entire query to avoid being stuck in intros/silence
        total_query_hashes = len(query_prints)
        print(f"🔍 Searching {total_query_hashes} query hashes (sampled)...")
        
        # Two-Pass matching with uniform sampling
        # Pass 1: 1000 hashes sampled uniformly
        # Pass 2: 4000 more hashes sampled uniformly
        passes = [1000, 4000]
        all_matches = defaultdict(list)
        
        # Global common hash cache (detect once and skip)
        if not hasattr(self, '_common_hashes'):
            self._common_hashes = set()
            # Relaxed threshold: 20,000 entries total (across 20 seasons)
            # This avoids filtering out hashes that are frequent in just one episode
            for h, arr in self.fingerprints.items():
                if len(arr) > 20000:
                    self._common_hashes.add(h)
            print(f"   - Filtered {len(self._common_hashes):,} extremely common hashes")

        queried_indices = set()
        
        for pass_idx, num_to_query in enumerate(passes):
            # Sample indices we haven't queried yet
            remaining_indices = [i for i in range(total_query_hashes) if i not in queried_indices]
            if not remaining_indices: break
            
            # Take a uniform sample
            sample_size = min(num_to_query, len(remaining_indices))
            step = max(1, len(remaining_indices) // sample_size)
            current_indices = remaining_indices[::step][:sample_size]
            queried_indices.update(current_indices)
            
            current_query = [query_prints[i] for i in current_indices]
            
            pass_matches = defaultdict(list)
            for h, t_q in current_query:
                if h in self.fingerprints and h not in self._common_hashes:
                    arr = self.fingerprints[h]
                    # Dynamic filtering (relaxed)
                    if len(arr) > 20000:
                        self._common_hashes.add(h)
                        continue
                        
                    for x in arr:
                        ep_idx = x['ep_idx']
                        t_db = x['offset']
                        ep_id = self.episode_list[ep_idx]
                        offset = t_db - t_q
                        all_matches[ep_id].append(offset)
                        pass_matches[ep_id].append(offset)
            
            # Check for strong match
            if pass_matches:
                best_ep, best_off, best_cnt, _ = self._find_best_alignment(pass_matches)
                # Strong match in Pass 1
                if pass_idx == 0 and best_cnt >= 30:
                    print(f"🚀 Strong match in Pass 1 ({best_cnt} aligned). Stopping.")
                    break
                # Decent match in Pass 2
                if pass_idx == 1 and best_cnt >= 25:
                    break

        if not all_matches: return {}
        
        best_episode, best_offset, best_count, top_candidates = self._find_best_alignment(all_matches)
        
        elapsed = time.time() - start_time
        print(f"⏱️  Matching took {elapsed:.2f}s")
        print("📊 Top 10 Candidates (Aligned Matches):")
        for c in top_candidates:
            print(f"   - {c['episode_id']}: {c['aligned_matches']} aligned ({c['raw_matches']} raw)")
        
        # Confidence calculation
        total_queried = len(queried_indices)
        min_aligned = max(10, total_queried * 0.01)
        if best_episode and best_count >= min_aligned:
            confidence = min(99, int((best_count / total_queried) * 1000))
            return {
                "episode_id": best_episode,
                "timestamp": max(0, best_offset),
                "confidence": confidence,
                "aligned_matches": best_count,
                "total_matches": len(all_matches[best_episode]),
                "method": "audio_pkl_optimized_v4",
                "processing_time": elapsed
            }
        return {}

    def _find_best_alignment(self, matches_dict: Dict[str, List[float]]) -> Tuple[str, float, int, List[Dict]]:
        """Find best alignment using binned voting. Returns best match and top 10 candidates."""
        best_ep, best_count, best_offset = None, 0, 0.0
        candidates = []
        
        # Increase candidate pool to 200 to avoid missing signal buried in noise
        sorted_candidates = sorted(matches_dict.items(), key=lambda x: len(x[1]), reverse=True)[:200]
        for ep_id, offsets in sorted_candidates:
            binned = [round(o * 2) / 2 for o in offsets]
            counts = Counter(binned)
            if not counts: continue
            mode_offset, count = counts.most_common(1)[0]
            
            candidates.append({
                "episode_id": ep_id,
                "aligned_matches": count,
                "raw_matches": len(offsets),
                "offset": mode_offset
            })
            
            if count > best_count:
                best_count, best_ep, best_offset = count, ep_id, mode_offset
        
        # Sort candidates by aligned matches
        candidates.sort(key=lambda x: x['aligned_matches'], reverse=True)
        return best_ep, best_offset, best_count, candidates[:10]

    def match_clip(self, file_path: str) -> Dict[str, Any]:
        try:
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            return self.match_audio_array(y, sr)
        except Exception as e:
            print(f"Error matching clip {file_path}: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        total_entries = sum(len(a) for a in self.fingerprints.values())
        return {
            "unique_hashes": len(self.fingerprints),
            "total_entries": total_entries,
            "episodes": len(self.episode_list),
            "avg_hashes_per_episode": total_entries / max(1, len(self.episode_list)),
            "loaded_databases": len(self.existing_dbs)
        }

# Global instance
audio_matcher = AudioFingerprinter()
