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
        
        # Per-season storage for parallel matching
        # season_num -> {hash -> numpy array of dtype=[('ep_idx', 'u2'), ('offset', 'f4')]}
        self.season_fingerprints: Dict[int, Dict[str, np.ndarray]] = {}
        # season_num -> [episode_ids]
        self.season_episodes: Dict[int, List[str]] = {}
        # season_num -> {ep_id -> ep_idx}
        self.season_episode_to_idx: Dict[int, Dict[str, int]] = {}

        # Track which databases are loaded (for lazy loading)
        self.loaded_seasons: set = set()

        # Number of parallel workers for season matching
        self.num_workers = int(os.getenv('MATCH_WORKERS', '16'))
        
        # OPTIMIZATION A: Incremental fingerprinting config
        self.incremental_durations = [3.0, 5.0, 8.0, 10.0]  # seconds
        self.min_aligned_early_exit = int(os.getenv('MIN_ALIGNED_EARLY_EXIT', '20'))
        
        # OPTIMIZATION B: Common hash filtering
        self.max_posting_list_size = int(os.getenv('MAX_POSTING_LIST_SIZE', '200'))
        self.common_hash_stoplist: set = set()
        
        # Check for lazy loading flag (default: False for backward compatibility)
        if lazy_load is None:
            lazy_load = os.getenv('LAZY_LOAD_PICKLE', 'false').lower() in ('true', '1', 'yes')
        self.lazy_load = lazy_load
        
        self._scan_databases()
        
        # OPTIMIZATION B: Load common hash stoplist if exists
        stoplist_path = os.path.join(self.data_dir, 'common_hash_stoplist_pkl.txt')
        if os.path.exists(stoplist_path):
            self._load_stoplist(stoplist_path)
            print(f"   ✓ Loaded {len(self.common_hash_stoplist):,} common hashes (stoplist)")
        
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

            # Initialize season storage
            self.season_fingerprints[season] = {}
            self.season_episodes[season] = []
            self.season_episode_to_idx[season] = {}

            for h, entries in raw_prints.items():
                # Convert entries for this hash to a small numpy array
                new_entries = []
                for ep_id, offset in entries:
                    if ep_id not in self.season_episode_to_idx[season]:
                        self.season_episode_to_idx[season][ep_id] = len(self.season_episodes[season])
                        self.season_episodes[season].append(ep_id)
                    ep_idx = self.season_episode_to_idx[season][ep_id]
                    new_entries.append((ep_idx, offset))

                new_arr = np.array(new_entries, dtype=dtype)
                self.season_fingerprints[season][h] = new_arr

            self.loaded_seasons.add(season)
            season_size = sum(a.nbytes for a in self.season_fingerprints[season].values()) / 1024 / 1024
            print(f"   ✓ Loaded {db_file} ({len(self.season_episodes[season])} episodes, {season_size:.1f}MB)")
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
        if self.season_fingerprints:
            total_eps = sum(len(eps) for eps in self.season_episodes.values())
            total_size = sum(
                sum(a.nbytes for a in fps.values())
                for fps in self.season_fingerprints.values()
            ) / 1024 / 1024
            print(f"✅ Loaded {len(self.loaded_seasons)} seasons ({total_eps} episodes) in {elapsed:.1f}s")
            print(f"📊 Total Data Size: {total_size:.1f}MB | {self.num_workers} parallel workers")

    def _load_stoplist(self, stoplist_path: str):
        """Load precomputed common hash stoplist from disk."""
        try:
            with open(stoplist_path, 'r') as f:
                self.common_hash_stoplist = {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"   ⚠️  Error loading stoplist: {e}")
            self.common_hash_stoplist = set()
    
    def build_stoplist(self, top_percent: float = 0.05, min_occurrences: int = 500):
        """
        OPTIMIZATION B: Build a stoplist of the most common hashes.
        
        Args:
            top_percent: Top X% of hashes by frequency to include
            min_occurrences: Minimum number of occurrences to be considered common
        """
        print(f"🔧 Building common hash stoplist (top {top_percent*100}%, min {min_occurrences} occurrences)...")
        
        # Ensure all databases are loaded
        if len(self.loaded_seasons) < len(self.existing_dbs):
            self._load_all_databases()
        
        hash_counts = Counter()
        
        # Count hash frequencies across all seasons
        for season, fingerprints in self.season_fingerprints.items():
            print(f"   Scanning season {season:02d}...")
            for h, arr in fingerprints.items():
                hash_counts[h] += len(arr)
        
        # Sort by frequency and select top X%
        total_hashes = len(hash_counts)
        sorted_hashes = hash_counts.most_common()
        
        # Calculate cutoff
        cutoff_count = int(total_hashes * top_percent)
        top_hashes = sorted_hashes[:cutoff_count]
        
        # Also filter by minimum occurrences
        common_hashes = {h for h, count in top_hashes if count >= min_occurrences}
        
        print(f"   ✓ Identified {len(common_hashes):,} common hashes")
        if sorted_hashes:
            print(f"   Example: hash '{sorted_hashes[0][0][:8]}...' appears {sorted_hashes[0][1]:,} times")
        
        # Save to disk
        stoplist_path = os.path.join(self.data_dir, 'common_hash_stoplist_pkl.txt')
        with open(stoplist_path, 'w') as f:
            for h in common_hashes:
                f.write(f"{h}\n")
        
        print(f"   ✓ Saved stoplist to {stoplist_path}")
        
        # Load into memory
        self.common_hash_stoplist = common_hashes
        
        return len(common_hashes)
    
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

    def _match_season(self, season: int, query_prints: List[Tuple[str, float]]) -> Dict[str, List[float]]:
        """
        Match query hashes against a single season's database.
        
        OPTIMIZATION B: Filter common hashes via stoplist + posting list cap.
        """
        season_matches = defaultdict(list)

        fingerprints = self.season_fingerprints.get(season, {})
        episode_list = self.season_episodes.get(season, [])

        if not fingerprints:
            return season_matches

        for h, t_q in query_prints:
            # OPTIMIZATION B: Skip if in stoplist
            if h in self.common_hash_stoplist:
                continue
            
            if h in fingerprints:
                arr = fingerprints[h]
                # OPTIMIZATION B: Cap posting list size
                if len(arr) > self.max_posting_list_size:
                    continue
                for x in arr:
                    ep_idx = x['ep_idx']
                    t_db = x['offset']
                    ep_id = episode_list[ep_idx]
                    offset = t_db - t_q
                    season_matches[ep_id].append(offset)

        return season_matches

    def match_audio_array(self, audio_array: np.ndarray, sr: int, incremental: bool = True) -> Dict[str, Any]:
        """
        Match audio array with OPTIMIZATIONS A+B+C.
        
        Args:
            audio_array: Audio time series
            sr: Sample rate
            incremental: If True, use incremental matching (default)
        """
        # Lazy load all databases if in lazy mode and not yet loaded
        if self.lazy_load and len(self.loaded_seasons) < len(self.existing_dbs):
            print("📦 Lazy loading databases for matching...")
            self._load_all_databases()

        start_time = time.time()
        
        if incremental:
            return self._match_incremental(audio_array, sr, start_time)
        else:
            return self._match_full(audio_array, sr, start_time)
    
    def _match_incremental(self, audio_array: np.ndarray, sr: int, start_time: float) -> Dict[str, Any]:
        """
        OPTIMIZATION A: Incremental fingerprinting.
        
        Start with 3 seconds, then extend to 5s, 8s, 10s.
        Stop as soon as we find a confident match.
        """
        print("🔍 Using INCREMENTAL matching (Optimization A+B+C)")
        
        duration = len(audio_array) / sr
        print(f"   Audio duration: {duration:.1f}s")
        
        # Incremental durations: 3s, 5s, 8s, 10s (or full duration)
        durations = [d for d in self.incremental_durations if d <= duration]
        if not durations or duration > self.incremental_durations[-1]:
            durations.append(duration)
        
        all_matches = defaultdict(list)
        total_fingerprints_generated = 0
        
        for step_idx, step_duration in enumerate(durations):
            step_start = time.time()
            
            # Extract audio chunk
            chunk_samples = int(step_duration * sr)
            audio_chunk = audio_array[:chunk_samples]
            
            print(f"\n📊 Step {step_idx+1}/{len(durations)}: Fingerprinting {step_duration:.1f}s...")
            
            # Generate fingerprints for this chunk
            peaks = self._get_spectrogram_peaks(audio_chunk)
            chunk_fingerprints = self._create_hashes(peaks)
            
            if not chunk_fingerprints:
                print("   No fingerprints found in chunk.")
                continue
            
            total_fingerprints_generated = len(chunk_fingerprints)
            print(f"   Generated {len(chunk_fingerprints)} hashes")
            
            # OPTIMIZATION B: Filter out common hashes
            filtered_hashes = [
                (h, t) for h, t in chunk_fingerprints 
                if h not in self.common_hash_stoplist
            ]
            
            filtered_count = len(chunk_fingerprints) - len(filtered_hashes)
            if filtered_count > 0:
                print(f"   Filtered {filtered_count} common hashes (stoplist)")
            
            if not filtered_hashes:
                print("   All hashes were common - extending...")
                continue
            
            # Search all seasons in parallel
            print(f"   Searching {len(filtered_hashes)} hashes across {len(self.loaded_seasons)} seasons...")
            lookup_start = time.time()
            
            step_matches = defaultdict(list)
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_season = {
                    executor.submit(self._match_season, season, filtered_hashes): season
                    for season in self.loaded_seasons
                }

                for future in as_completed(future_to_season):
                    try:
                        season_matches = future.result()
                        for ep_id, offsets in season_matches.items():
                            all_matches[ep_id].extend(offsets)
                            step_matches[ep_id].extend(offsets)
                    except Exception as e:
                        pass  # Silent fail for individual seasons
            
            lookup_time = time.time() - lookup_start
            print(f"   ⏱️  Lookup: {lookup_time:.2f}s")
            
            # Check if we have a confident match
            if step_matches:
                best_ep, best_off, best_cnt, _ = self._find_best_alignment(step_matches)
                print(f"   🎯 Best match: {best_ep} ({best_cnt} aligned hashes)")
                
                # OPTIMIZATION A: Early exit if confident
                if best_cnt >= self.min_aligned_early_exit:
                    elapsed = time.time() - start_time
                    print(f"\n🚀 CONFIDENT MATCH found at step {step_idx+1}/{len(durations)}")
                    print(f"   Only fingerprinted {step_duration:.1f}s (vs {duration:.1f}s full)")
                    print(f"   Total time: {elapsed:.2f}s")
                    
                    confidence = min(99, int((best_cnt / len(chunk_fingerprints)) * 1000))
                    
                    return {
                        "episode_id": best_ep,
                        "timestamp": max(0, best_off),
                        "confidence": confidence,
                        "aligned_matches": best_cnt,
                        "total_matches": len(all_matches[best_ep]),
                        "method": f"incremental_pkl_step_{step_idx+1}",
                        "processing_time": elapsed,
                        "duration_used": step_duration,
                        "duration_saved": duration - step_duration
                    }
            
            step_time = time.time() - step_start
            print(f"   Step {step_idx+1} total: {step_time:.2f}s")
        
        # No confident match found after all steps
        print(f"\n⚠️  No confident match after all incremental steps")
        
        if all_matches:
            best_episode, best_offset, best_count, top_candidates = self._find_best_alignment(all_matches)
            elapsed = time.time() - start_time
            
            print(f"⏱️  Total matching took {elapsed:.2f}s")
            print("📊 Top 10 Candidates:")
            for c in top_candidates[:10]:
                print(f"   - {c['episode_id']}: {c['aligned_matches']} aligned")
            
            # Return best match even if below threshold
            if best_count >= 5:
                confidence = int((best_count / total_fingerprints_generated) * 100) if total_fingerprints_generated > 0 else 0
                return {
                    "episode_id": best_episode,
                    "timestamp": max(0, best_offset),
                    "confidence": confidence,
                    "aligned_matches": best_count,
                    "total_matches": len(all_matches[best_episode]),
                    "method": "incremental_pkl_weak",
                    "processing_time": elapsed
                }
        
        return {}
    
    def _match_full(self, audio_array: np.ndarray, sr: int, start_time: float) -> Dict[str, Any]:
        """
        Full audio matching (non-incremental, for backward compatibility).
        Still uses optimization B (common hash filtering).
        """
        print("🔍 Using FULL matching (Optimization B+C, no incremental)")
        
        peaks = self._get_spectrogram_peaks(audio_array)
        query_prints = self._create_hashes(peaks)
        if not query_prints:
            return {}

        total_query_hashes = len(query_prints)
        print(f"   Generated {total_query_hashes} hashes")
        
        # OPTIMIZATION B: Filter common hashes
        filtered_hashes = [
            (h, t) for h, t in query_prints 
            if h not in self.common_hash_stoplist
        ]
        
        filtered_count = len(query_prints) - len(filtered_hashes)
        if filtered_count > 0:
            print(f"   Filtered {filtered_count} common hashes (stoplist)")
        
        if not filtered_hashes:
            print("   All hashes were common.")
            return {}
        
        print(f"🔍 Searching {len(filtered_hashes)} hashes across {len(self.loaded_seasons)} seasons...")

        # Two-pass strategy with parallel season search
        passes = [1000, 3000]
        all_matches = defaultdict(list)
        queried_indices = set()

        for pass_idx, num_to_query in enumerate(passes):
            # Sample indices we haven't queried yet
            remaining_indices = [i for i in range(len(filtered_hashes)) if i not in queried_indices]
            if not remaining_indices:
                break

            sample_size = min(num_to_query, len(remaining_indices))
            step = max(1, len(remaining_indices) // sample_size)
            current_indices = remaining_indices[::step][:sample_size]
            queried_indices.update(current_indices)

            current_query = [filtered_hashes[i] for i in current_indices]

            # Search all seasons in parallel
            pass_matches = defaultdict(list)

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_season = {
                    executor.submit(self._match_season, season, current_query): season
                    for season in self.loaded_seasons
                }

                for future in as_completed(future_to_season):
                    try:
                        season_matches = future.result()
                        for ep_id, offsets in season_matches.items():
                            all_matches[ep_id].extend(offsets)
                            pass_matches[ep_id].extend(offsets)
                    except Exception as e:
                        pass  # Silent fail for individual seasons

            # Check for strong match and early exit
            if pass_matches:
                best_ep, best_off, best_cnt, _ = self._find_best_alignment(pass_matches)
                if pass_idx == 0 and best_cnt >= 30:
                    print(f"🚀 Strong match in Pass 1 ({best_cnt} aligned). Stopping early.")
                    break
                if pass_idx == 1 and best_cnt >= 25:
                    break

        if not all_matches:
            print("   ❌ No matches found")
            return {}

        # Find best match
        best_episode, best_offset, best_count, top_candidates = self._find_best_alignment(all_matches)

        elapsed = time.time() - start_time
        print(f"⏱️  Parallel matching took {elapsed:.2f}s")
        print("📊 Top 10 Candidates:")
        for c in top_candidates[:10]:
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
                "method": "audio_pkl_full",
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
        total_episodes = sum(len(eps) for eps in self.season_episodes.values())
        total_hashes = sum(len(fps) for fps in self.season_fingerprints.values())
        total_entries = sum(
            sum(len(a) for a in fps.values())
            for fps in self.season_fingerprints.values()
        )
        return {
            "unique_hashes": total_hashes,
            "total_entries": total_entries,
            "episodes": total_episodes,
            "seasons_loaded": len(self.loaded_seasons),
            "existing_databases": len(self.existing_dbs),
            "lazy_load_mode": self.lazy_load,
            "parallel_workers": self.num_workers
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

        # Initialize season storage if needed
        if season not in self.season_fingerprints:
            self.season_fingerprints[season] = {}
            self.season_episodes[season] = []
            self.season_episode_to_idx[season] = {}

        # Add episode to season's episode list if not present
        if episode_id not in self.season_episode_to_idx[season]:
            self.season_episode_to_idx[season][episode_id] = len(self.season_episodes[season])
            self.season_episodes[season].append(episode_id)

        ep_idx = self.season_episode_to_idx[season][episode_id]
        dtype = [('ep_idx', 'u2'), ('offset', 'f4')]

        # Group fingerprints by hash and add to season's in-memory database
        for h, offset in fingerprints:
            new_entry = np.array([(ep_idx, offset)], dtype=dtype)

            if h in self.season_fingerprints[season]:
                self.season_fingerprints[season][h] = np.concatenate([self.season_fingerprints[season][h], new_entry])
            else:
                self.season_fingerprints[season][h] = new_entry

        print(f"✓ Added {episode_id}: {len(fingerprints)} fingerprints")
        return True
    
    def force_save_all(self):
        """Save all in-memory fingerprints to pickle files, organized by season."""
        print("💾 Saving all databases to pickle files...")

        # Save each season database
        saved_count = 0
        for season in self.season_fingerprints.keys():
            # Convert numpy arrays back to list format for pickling
            fingerprints_to_save = defaultdict(list)
            episode_list = self.season_episodes[season]

            for h, arr in self.season_fingerprints[season].items():
                for entry in arr:
                    ep_idx = entry['ep_idx']
                    offset = entry['offset']
                    ep_id = episode_list[ep_idx]
                    fingerprints_to_save[h].append((ep_id, offset))

            fingerprints = fingerprints_to_save
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
        self.season_fingerprints.clear()
        self.season_episodes.clear()
        self.season_episode_to_idx.clear()
        self.loaded_seasons.clear()
        print("🗑️  Cleared all fingerprints from memory")

# Global instance - use lazy loading if environment variable is set
audio_matcher = AudioFingerprinter()
