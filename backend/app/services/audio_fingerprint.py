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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    
    def __init__(self, lazy_load: bool = None):
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
        
        # Track which databases are loaded (for lazy loading)
        self.loaded_seasons: set = set()
        
        # Check for lazy loading flag (default: False for backward compatibility)
        if lazy_load is None:
            lazy_load = os.getenv('LAZY_LOAD_PICKLE', 'false').lower() in ('true', '1', 'yes')
        self.lazy_load = lazy_load
        
        self._scan_databases()
        
        if self.lazy_load:
            print(f"📦 Lazy loading mode: Skipping initial database load (will load on-demand)")
            print(f"   Found {len(self.existing_dbs)} existing databases")
        else:
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

    def _load_database(self, season: int) -> bool:
        """Load a single database file for a season. Returns True if loaded successfully."""
        if season in self.loaded_seasons:
            return True  # Already loaded
        
        db_file = self.DB_CONFIG.get(season)
        if not db_file:
            return False
        
        db_path = os.path.join(self.data_dir, db_file)
        if not os.path.exists(db_path):
            return False
        
        try:
            print(f"📂 Loading {db_file}...")
            dtype = [('ep_idx', 'u2'), ('offset', 'f4')]
            
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
            
            self.loaded_seasons.add(season)
            print(f"   ✓ Loaded {db_file} (RAM: {sum(a.nbytes for a in self.fingerprints.values()) / 1024 / 1024:.1f}MB)")
            del data  # Explicitly free memory
            return True
        except Exception as e:
            print(f"   ❌ Error loading {db_file}: {e}")
            return False
    
    def _load_all_databases(self):
        """Load all existing databases into memory."""
        start_time = time.time()
        
        for db_file in self.existing_dbs:
            # Extract season number from filename
            match = re.search(r's(\d+)', db_file)
            if match:
                season = int(match.group(1))
                self._load_database(season)

        elapsed = time.time() - start_time
        if self.fingerprints:
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
        # Lazy load all databases if in lazy mode and not yet loaded
        if self.lazy_load and len(self.loaded_seasons) < len(self.existing_dbs):
            print("📦 Lazy loading databases for matching...")
            self._load_all_databases()
        
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
            
            # OPTIMIZATION: Parallelize hash matching using threading (I/O-bound operation)
            pass_matches = defaultdict(list)
            
            num_workers = min(8, len(current_query) // 100)  # Use threads for chunks of 100+ hashes
            chunk_size = max(100, len(current_query) // num_workers) if num_workers > 1 else len(current_query)
            
            def match_hash_chunk(chunk):
                """Match a chunk of query hashes against database (thread-safe)."""
                chunk_matches = defaultdict(list)
                for h, t_q in chunk:
                    if h in self.fingerprints and h not in self._common_hashes:
                        arr = self.fingerprints[h]
                        # Dynamic filtering (relaxed)
                        if len(arr) > 20000:
                            # Note: Adding to common_hashes is not thread-safe, but acceptable for performance
                            continue
                        for x in arr:
                            ep_idx = x['ep_idx']
                            t_db = x['offset']
                            ep_id = self.episode_list[ep_idx]
                            offset = t_db - t_q
                            chunk_matches[ep_id].append(offset)
                return chunk_matches
            
            # Split query into chunks and process in parallel
            if len(current_query) > 200 and num_workers > 1:
                query_chunks = [current_query[i:i+chunk_size] for i in range(0, len(current_query), chunk_size)]
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(match_hash_chunk, chunk) for chunk in query_chunks]
                    for future in as_completed(futures):
                        chunk_result = future.result()
                        for ep_id, offsets in chunk_result.items():
                            all_matches[ep_id].extend(offsets)
                            pass_matches[ep_id].extend(offsets)
            else:
                # Sequential processing for small queries
                for h, t_q in current_query:
                    if h in self.fingerprints and h not in self._common_hashes:
                        arr = self.fingerprints[h]
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
            "loaded_databases": len(self.loaded_seasons),
            "existing_databases": len(self.existing_dbs),
            "lazy_load_mode": self.lazy_load
        }
    
    def _get_season_from_episode_id(self, episode_id: str) -> Optional[int]:
        """Extract season number from episode ID (e.g., 'S01E05' -> 1)."""
        match = re.search(r'[Ss](\d+)', episode_id)
        if match:
            return int(match.group(1))
        return None
    
    def fingerprint_audio_array(self, audio_array: np.ndarray, sr: int) -> List[Tuple[str, float]]:
        """Generate fingerprints from audio array. Returns list of (hash, offset) tuples."""
        peaks = self._get_spectrogram_peaks(audio_array)
        return self._create_hashes(peaks)
    
    def add_episode(self, episode_id: str, file_path: str) -> bool:
        """
        Add an episode to the database.
        
        Args:
            episode_id: Episode identifier (e.g., 'S04E05')
            file_path: Path to audio/video file
            
        Returns:
            True if successful
        """
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            return self.add_episode_array(episode_id, y, sr)
        except Exception as e:
            print(f"Error adding episode {episode_id}: {e}")
            return False
    
    def add_episode_array(self, episode_id: str, audio_array: np.ndarray, sr: int) -> bool:
        """
        Add an episode from audio array.
        
        Args:
            episode_id: Episode identifier (e.g., 'S04E05')
            audio_array: Audio time series
            sr: Sample rate
            
        Returns:
            True if successful
        """
        # Generate fingerprints
        fingerprints = self.fingerprint_audio_array(audio_array, sr)
        
        if not fingerprints:
            print(f"⚠️  No fingerprints generated for {episode_id}")
            return False
        
        # Determine season
        season = self._get_season_from_episode_id(episode_id)
        if season is None:
            print(f"⚠️  Cannot determine season from episode ID: {episode_id}")
            return False
        
        # Load the season database if not already loaded (for merging)
        if season not in self.loaded_seasons:
            self._load_database(season)
        
        # Add episode to episode list if not present
        if episode_id not in self.episode_to_idx:
            self.episode_to_idx[episode_id] = len(self.episode_list)
            self.episode_list.append(episode_id)
        
        ep_idx = self.episode_to_idx[episode_id]
        dtype = [('ep_idx', 'u2'), ('offset', 'f4')]
        
        # Group fingerprints by hash and add to in-memory database
        for h, offset in fingerprints:
            new_entry = np.array([(ep_idx, offset)], dtype=dtype)
            
            if h in self.fingerprints:
                self.fingerprints[h] = np.concatenate([self.fingerprints[h], new_entry])
            else:
                self.fingerprints[h] = new_entry
        
        print(f"✓ Added {episode_id}: {len(fingerprints)} fingerprints")
        return True
    
    def force_save_all(self):
        """Save all in-memory fingerprints to pickle files, organized by season."""
        print("💾 Saving all databases to pickle files...")
        
        # Group fingerprints by season
        season_fingerprints: Dict[int, Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
        
        # Convert in-memory format back to per-season format
        for h, arr in self.fingerprints.items():
            for entry in arr:
                ep_idx = entry['ep_idx']
                offset = entry['offset']
                ep_id = self.episode_list[ep_idx]
                season = self._get_season_from_episode_id(ep_id)
                if season:
                    season_fingerprints[season][h].append((ep_id, offset))
        
        # Save each season database
        saved_count = 0
        for season, fingerprints in season_fingerprints.items():
            db_file = self.DB_CONFIG.get(season)
            if not db_file:
                continue
            
            db_path = os.path.join(self.data_dir, db_file)
            
            # Load existing database if it exists and merge
            existing_fingerprints = {}
            if os.path.exists(db_path):
                try:
                    with open(db_path, 'rb') as f:
                        existing_data = pickle.load(f)
                    existing_fingerprints = existing_data.get('fingerprints', existing_data) if isinstance(existing_data, dict) else existing_data
                except Exception as e:
                    print(f"   ⚠️  Could not load existing {db_file}: {e}")
            
            # Merge with existing
            for h, entries in existing_fingerprints.items():
                if h not in fingerprints:
                    fingerprints[h] = entries
                else:
                    # Merge entries, avoiding duplicates
                    existing_ep_offsets = set((ep_id, round(offset, 2)) for ep_id, offset in fingerprints[h])
                    for ep_id, offset in entries:
                        if (ep_id, round(offset, 2)) not in existing_ep_offsets:
                            fingerprints[h].append((ep_id, offset))
            
            # Count episodes in this season
            episode_counts = {}
            for h, entries in fingerprints.items():
                for ep_id, _ in entries:
                    if ep_id not in episode_counts:
                        episode_counts[ep_id] = 0
                    episode_counts[ep_id] += 1
            
            # Save to pickle file
            try:
                save_data = {
                    'fingerprints': dict(fingerprints),
                    'counts': episode_counts,
                    'season': season
                }
                
                with open(db_path, 'wb') as f:
                    pickle.dump(save_data, f)
                
                print(f"   ✓ Saved {db_file} ({len(fingerprints):,} hashes, {len(episode_counts)} episodes)")
                saved_count += 1
            except Exception as e:
                print(f"   ❌ Error saving {db_file}: {e}")
        
        print(f"✅ Saved {saved_count} database files")
    
    def clear_db(self):
        """Clear all in-memory fingerprints (for testing)."""
        self.fingerprints.clear()
        self.episode_list.clear()
        self.episode_to_idx.clear()
        self.loaded_seasons.clear()
        print("🗑️  Cleared all fingerprints from memory")

# Global instance - use lazy loading if environment variable is set
audio_matcher = AudioFingerprinter()
