"""Audio Fingerprinting Service - Shazam-style spectral peak landmark fingerprinting"""

import os
import re
import numpy as np
import librosa
import pickle
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from scipy.ndimage import maximum_filter
from app.core.config import settings


class AudioFingerprinter:
    """Shazam-style audio fingerprinting with spectral peak landmarks."""
    
    # Database configuration: season_number -> database_filename
    # Each season gets its own database file (audio_fingerprints_sXX.pkl)
    DB_CONFIG = {
        season: f"audio_fingerprints_s{season:02d}.pkl"
        for season in range(1, 21)  # Seasons 1-20
    }
    
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        
        # Limit seasons to load (for testing/deployment)
        # Set MAX_SEASONS environment variable to limit (e.g., MAX_SEASONS=1 for season 1 only)
        self.max_seasons = int(os.getenv('MAX_SEASONS', '20'))  # Default to all 20 seasons
        
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
        
        # Multi-database storage: db_file -> {fingerprints, episode_hash_counts}
        # Only loaded databases are kept in memory (lazy loading)
        self.loaded_dbs: Dict[str, Dict[str, Any]] = {}
        
        # Batch saving: track unsaved changes to avoid O(n^2) save operations
        self.unsaved_episodes: Dict[str, int] = {}  # db_file -> count of unsaved episodes
        self.save_batch_size = 5  # Save every N episodes (reduces O(n^2) to O(n))
        
        # Track which database files exist
        self._scan_databases()
        
        # Eagerly load all databases at startup for instant responses
        EAGER_LOAD = os.getenv('EAGER_LOAD_DB', 'true').lower() in ('true', '1', 'yes')
        if EAGER_LOAD:
            print(f"📦 Eager loading all databases into memory at startup...")
            self._load_all_databases()
        else:
            print(f"   ℹ️  Lazy loading enabled - databases will load on-demand")

        print(f"   (Batch saving: every {self.save_batch_size} episodes to optimize performance)")
        if self.max_seasons < 20:
            print(f"   ⚠️  MAX_SEASONS={self.max_seasons} - only loading seasons 1-{self.max_seasons}")
        
    def _scan_databases(self):
        """Scan for existing database files, respecting MAX_SEASONS limit."""
        self.existing_dbs = set()
        for season in range(1, min(self.max_seasons + 1, 21)):  # Limit to max_seasons
            db_file = self.DB_CONFIG.get(season)
            if db_file:
                db_path = os.path.join(self.data_dir, db_file)
                if os.path.exists(db_path):
                    self.existing_dbs.add(db_file)

    def _load_all_databases(self):
        """Eagerly load all existing database files at startup."""
        if not self.existing_dbs:
            print("   ℹ️  No existing database files found")
            return

        print(f"📦 Loading {len(self.existing_dbs)} season database(s) at startup...")
        for db_file in self.existing_dbs:
            self._load_db_file(db_file)
        print(f"   ✓ Loaded {len(self.loaded_dbs)} database file(s)")

    def _get_all_db_files(self) -> List[str]:
        """Get list of all configured database files."""
        return list(self.DB_CONFIG.values())
    
    def _get_season_from_episode_id(self, episode_id: str) -> Optional[int]:
        """Extract season number from episode ID (e.g., 'S01E05' -> 1)."""
        match = re.search(r'[Ss](\d+)', episode_id)
        if match:
            return int(match.group(1))
        return None
    
    def _get_db_file_for_season(self, season: int) -> str:
        """Determine which database file to use for a given season."""
        if season in self.DB_CONFIG:
            return self.DB_CONFIG[season]
        # Default to season 1 database if season doesn't match
        return self.DB_CONFIG.get(1, "audio_fingerprints_s01.pkl")
    
    def _get_db_file_for_episode(self, episode_id: str) -> str:
        """Determine which database file to use for an episode."""
        season = self._get_season_from_episode_id(episode_id)
        if season is None:
            # Default to first database if we can't determine season
            return list(self.DB_CONFIG.values())[0]
        return self._get_db_file_for_season(season)
    
    def _load_db_file(self, db_file: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a specific database file (lazy loading).
        Returns the database data structure.
        """
        if db_file in self.loaded_dbs and not force_reload:
            return self.loaded_dbs[db_file]
        
        db_path = os.path.join(self.data_dir, db_file)
        
        if not os.path.exists(db_path):
            # Create empty database structure
            db_data = {
                'fingerprints': {},
                'counts': {}
            }
            self.loaded_dbs[db_file] = db_data
            return db_data
        
        try:
            print(f"📂 Loading database: {db_file}...")
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'fingerprints' in data:
                db_data = {
                    'fingerprints': data['fingerprints'],
                    'counts': data.get('counts', {})
                }
            else:
                # Legacy format
                db_data = {
                    'fingerprints': data,
                    'counts': {}
                }
            
            self.loaded_dbs[db_file] = db_data
            print(f"   ✓ Loaded {len(db_data['fingerprints'])} unique hashes from {db_file}")
            return db_data
            
        except Exception as e:
            print(f"   ❌ Error loading {db_file}: {e}")
            db_data = {'fingerprints': {}, 'counts': {}}
            self.loaded_dbs[db_file] = db_data
            return db_data
    
    def _save_db_file(self, db_file: str):
        """Save a specific database file to disk."""
        if db_file not in self.loaded_dbs:
            return
        
        db_data = self.loaded_dbs[db_file]
        db_path = os.path.join(self.data_dir, db_file)
        
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            # Write to temp file first, then rename atomically
            temp_path = db_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    'fingerprints': db_data['fingerprints'],
                    'counts': db_data['counts']
                }, f)
            # Atomic rename (prevents corruption if interrupted)
            os.replace(temp_path, db_path)
            print(f"   ✓ Saved {len(db_data['fingerprints'])} hashes, {len(db_data['counts'])} episodes to {db_file}")
        except Exception as e:
            print(f"   ❌ Error saving {db_file}: {e}")
            # Clean up temp file if it exists
            temp_path = db_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def force_save_all(self):
        """Force save all databases (useful at end of ingestion)."""
        print("💾 Force saving all databases...")
        for db_file in self.loaded_dbs.keys():
            self._save_db_file(db_file)
        self.unsaved_episodes = {}
        print("✓ All databases saved")
    
    def _get_all_fingerprints(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get combined fingerprints from all databases.
        Loads databases on-demand if not already loaded (lazy loading).
        Respects MAX_SEASONS limit.
        """
        # Load all existing databases if not already loaded (respecting MAX_SEASONS)
        for db_file in self.existing_dbs:
            if db_file not in self.loaded_dbs:
                self._load_db_file(db_file)
        
        combined = {}
        for db_file, db_data in self.loaded_dbs.items():
            fingerprints = db_data['fingerprints']
            # Merge fingerprints (hash -> list of (episode_id, time))
            for hash_val, entries in fingerprints.items():
                if hash_val not in combined:
                    combined[hash_val] = []
                combined[hash_val].extend(entries)
        return combined
    
    def clear_db(self, db_file: Optional[str] = None):
        """Clear fingerprints from a specific database or all databases."""
        if db_file:
            if db_file in self.loaded_dbs:
                self.loaded_dbs[db_file] = {'fingerprints': {}, 'counts': {}}
            db_path = os.path.join(self.data_dir, db_file)
            if os.path.exists(db_path):
                os.remove(db_path)
            print(f"✓ Cleared database: {db_file}")
        else:
            # Clear all
            self.loaded_dbs = {}
            for db_file in self._get_all_db_files():
                db_path = os.path.join(self.data_dir, db_file)
                if os.path.exists(db_path):
                    os.remove(db_path)
            print("✓ Cleared all audio databases")

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
        """Process an episode and store its fingerprints in the appropriate database."""
        fingerprints = self.fingerprint_audio(file_path)
        
        if not fingerprints:
            print(f"⚠️ No audio fingerprints generated for {episode_id}")
            return
        
        # Determine which database file to use
        db_file = self._get_db_file_for_episode(episode_id)
        db_data = self._load_db_file(db_file)
        
        # Add to database
        count = 0
        for h, offset in fingerprints:
            if h not in db_data['fingerprints']:
                db_data['fingerprints'][h] = []
            db_data['fingerprints'][h].append((episode_id, offset))
            count += 1
        
        db_data['counts'][episode_id] = count
        print(f"✓ Added {count} audio hashes for {episode_id} to {db_file}")
        
        # Batch saving: only save every N episodes to avoid O(n^2) complexity
        # This dramatically speeds up ingestion as the database grows
        self.unsaved_episodes[db_file] = self.unsaved_episodes.get(db_file, 0) + 1
        
        if self.unsaved_episodes[db_file] >= self.save_batch_size:
            print(f"   💾 Batch save triggered ({self.unsaved_episodes[db_file]} episodes)")
            self._save_db_file(db_file)
            self.unsaved_episodes[db_file] = 0
        else:
            print(f"   ⏳ Deferred save ({self.unsaved_episodes[db_file]}/{self.save_batch_size} episodes)")

    def match_audio_array(self, audio_array: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Match an audio array against all databases.
        Same as match_clip but works with audio array directly (no file I/O).
        """
        print("🔍 Fingerprinting query clip from array...")
        
        # Generate fingerprints from array
        peaks = self._get_spectrogram_peaks(audio_array)
        query_prints = self._create_hashes(peaks)
        
        if not query_prints:
            print("No fingerprints found in query.")
            return {}
        
        print(f"   Generated {len(query_prints)} query fingerprints")
        
        # Load all databases and combine fingerprints for matching
        print("📂 Loading databases for matching...")
        all_fingerprints = self._get_all_fingerprints()
        
        if not all_fingerprints:
            print("Audio database is empty.")
            return {}
        
        print(f"   Searching across {len(self.loaded_dbs)} loaded database(s)")
        print(f"   Database has {len(all_fingerprints):,} unique hashes")
        
        # Check how many query hashes exist in database
        found_in_db = sum(1 for h, _ in query_prints if h in all_fingerprints)
        print(f"   Query hashes found in DB: {found_in_db}/{len(query_prints)} ({found_in_db/len(query_prints)*100:.1f}%)")
        
        # OPTIMIZATION 1: Filter out overly common hashes
        # Use percentage-based threshold: filter hashes in >30% of episodes
        # This scales properly whether we have 47 episodes (seasons 1-3) or 271 episodes (all seasons)
        total_episodes = len(set(ep for episodes in all_fingerprints.values() for ep, _ in episodes))
        max_episodes_per_hash = max(100, int(total_episodes * 0.3))  # At least 100, or 30% of total
        print(f"   Filtering hashes appearing in >{max_episodes_per_hash} episodes (30% of {total_episodes} total)")
        filtered_prints = []
        skipped_common = 0
        for h, t_query in query_prints:
            if h in all_fingerprints:
                if len(all_fingerprints[h]) > max_episodes_per_hash:
                    skipped_common += 1
                    continue
                filtered_prints.append((h, t_query))
        
        if skipped_common > 0:
            print(f"⏭️  Skipped {skipped_common} overly common hashes (appear in >{max_episodes_per_hash} episodes)")
        
        print(f"   After filtering: {len(filtered_prints)} query hashes to search")
        
        # Sample if too many
        max_query_hashes = 10000
        if len(filtered_prints) > max_query_hashes:
            step = len(filtered_prints) // max_query_hashes
            filtered_prints = filtered_prints[::step]
            print(f"📊 Sampling: Using {len(filtered_prints)} of {len(query_prints)} query hashes")
        
        print(f"Searching {len(filtered_prints)} query hashes against {len(all_fingerprints)} DB hashes...")
        
        # Find matches
        matches: Dict[str, List[float]] = defaultdict(list)
        match_count = 0
        
        for h, t_query in filtered_prints:
            if h in all_fingerprints:
                for ep_id, t_db in all_fingerprints[h]:
                    offset = t_db - t_query
                    matches[ep_id].append(offset)
                    match_count += 1
        
        print(f"Found {match_count} raw hash matches across {len(matches)} episodes")
        
        if not matches:
            return {}
        
        # Find best match using time-aligned voting
        best_episode = None
        best_count = 0
        best_offset = 0.0
        
        sorted_episodes = sorted(matches.items(), key=lambda x: len(x[1]), reverse=True)
        
        for ep_id, offsets in sorted_episodes:
            if len(offsets) < 5:
                continue
            
            if len(offsets) > 5000:
                offsets = offsets[:5000]
            
            binned = [round(o * 2) / 2 for o in offsets]
            counts = Counter(binned)
            
            if not counts:
                continue
            
            mode_offset, count = counts.most_common(1)[0]
            
            if count > best_count:
                best_count = count
                best_episode = ep_id
                best_offset = mode_offset
            
            # Early termination
            if best_count > 100 and len(sorted_episodes) > 1:
                next_ep_id, next_offsets = sorted_episodes[1]
                if len(next_offsets) < best_count / 3:
                    print(f"✓ Early termination: Clear winner found ({best_count} vs max {len(next_offsets)})")
                    break
        
        print(f"Best match: {best_episode} with {best_count} aligned hashes at offset {best_offset:.1f}s")
        
        # Require minimum aligned matches (same logic as match_clip)
        duration_seconds = len(filtered_prints) / 30  # Rough estimate: ~30 hashes per second
        if duration_seconds < 120:  # Less than 2 minutes
            min_aligned = max(5, len(filtered_prints) * 0.005)  # 0.5% or 5 matches
        else:
            min_aligned = max(10, len(filtered_prints) * 0.01)  # 1% or 10 matches
        
        if best_episode and best_count >= min_aligned:
            # Calculate base confidence from alignment ratio
            base_confidence = (best_count / len(filtered_prints)) * 100

            # Boost confidence for strong matches (many aligned hashes)
            # This accounts for TikTok's re-encoding which reduces alignment %
            if best_count >= 100:
                # Very strong match: boost significantly
                confidence = min(95, int(base_confidence * 3))
            elif best_count >= 50:
                # Strong match: boost moderately
                confidence = min(85, int(base_confidence * 2.5))
            elif best_count >= 20:
                # Moderate match: boost slightly
                confidence = min(75, int(base_confidence * 2))
            else:
                # Weak match: use base confidence
                confidence = min(65, int(base_confidence * 1.5))

            return {
                "episode_id": best_episode,
                "timestamp": max(0, best_offset),
                "confidence": confidence,
                "aligned_matches": best_count,
                "total_matches": match_count,
                "method": "audio"
            }
        
        return {}

    def match_clip(self, file_path: str) -> Dict[str, Any]:
        """
        Match a query clip against all databases.
        
        Uses time-aligned voting: only counts matches where
        multiple hashes agree on the same time offset.
        This eliminates false positives from coincidental matches.
        
        OPTIMIZED: Skips common hashes and uses early termination.
        Databases are loaded on-demand during matching.
        """
        print("🔍 Fingerprinting query clip...")
        query_prints = self.fingerprint_audio(file_path)
        
        if not query_prints:
            print("No fingerprints found in query.")
            return {}
        
        print(f"   Generated {len(query_prints)} query fingerprints")
        
        # Load all databases and combine fingerprints for matching
        # This is done on-demand (lazy loading)
        print("📂 Loading databases for matching...")
        all_fingerprints = self._get_all_fingerprints()
        
        if not all_fingerprints:
            print("Audio database is empty.")
            return {}
        
        print(f"   Searching across {len(self.loaded_dbs)} loaded database(s)")
        print(f"   Database has {len(all_fingerprints):,} unique hashes")
        
        # Check how many query hashes exist in database
        found_in_db = sum(1 for h, _ in query_prints if h in all_fingerprints)
        print(f"   Query hashes found in DB: {found_in_db}/{len(query_prints)} ({found_in_db/len(query_prints)*100:.1f}%)")
        
        # Filter out overly common hashes
        # Use percentage-based threshold: filter hashes in >50% of episodes
        # This scales properly whether we have 47 episodes (seasons 1-3) or 271 episodes (all seasons)
        total_episodes = len(set(ep for episodes in all_fingerprints.values() for ep, _ in episodes))
        max_episodes_per_hash = max(100, int(total_episodes * 0.5))  # At least 100, or 50% of total
        print(f"   Filtering hashes appearing in >{max_episodes_per_hash} episodes (50% of {total_episodes} total)")
        filtered_prints = []
        skipped_common = 0
        for h, t_query in query_prints:
            if h in all_fingerprints:
                # Skip hashes that appear in too many episodes (likely common sounds)
                if len(all_fingerprints[h]) > max_episodes_per_hash:
                    skipped_common += 1
                    continue
                filtered_prints.append((h, t_query))
        
        if skipped_common > 0:
            print(f"⏭️  Skipped {skipped_common} overly common hashes (appear in >{max_episodes_per_hash} episodes)")
        
        print(f"   After filtering: {len(filtered_prints)} query hashes to search")
        
        # Sample query hashes if too many
        max_query_hashes = 10000
        if len(filtered_prints) > max_query_hashes:
            step = len(filtered_prints) // max_query_hashes
            filtered_prints = filtered_prints[::step]
            print(f"📊 Sampling: Using {len(filtered_prints)} of {len(query_prints)} query hashes")
        
        print(f"Searching {len(filtered_prints)} query hashes against {len(all_fingerprints)} DB hashes...")
        
        # Find matches: episode -> list of (db_time - query_time) offsets
        matches: Dict[str, List[float]] = defaultdict(list)
        match_count = 0
        
        for h, t_query in filtered_prints:
            if h in all_fingerprints:
                for ep_id, t_db in all_fingerprints[h]:
                    offset = t_db - t_query
                    matches[ep_id].append(offset)
                    match_count += 1
        
        print(f"Found {match_count} raw hash matches across {len(matches)} episodes")
        
        if not matches:
            return {}
        
        # Find best match using time-aligned voting
        # Early termination if clear winner found
        best_episode = None
        best_count = 0
        best_offset = 0.0
        
        # Sort episodes by match count (most likely first)
        sorted_episodes = sorted(matches.items(), key=lambda x: len(x[1]), reverse=True)
        
        for ep_id, offsets in sorted_episodes:
            if len(offsets) < 5:  # Need minimum matches
                continue
            
            # Limit offset processing (first 5000 per episode)
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
            
            # Early termination if clear winner (3x better than next)
            if best_count > 100 and len(sorted_episodes) > 1:
                # Check if next episode could beat us
                next_ep_id, next_offsets = sorted_episodes[1]
                if len(next_offsets) < best_count / 3:
                    print(f"✓ Early termination: Clear winner found ({best_count} vs max {len(next_offsets)})")
                    break
        
        print(f"Best match: {best_episode} with {best_count} aligned hashes at offset {best_offset:.1f}s")
        
        # Require minimum aligned matches to avoid false positives
        # For short clips (< 2 min), use lower threshold as they have fewer hashes
        # For longer clips, require more alignment
        duration_seconds = len(filtered_prints) / 30  # Rough estimate: ~30 hashes per second
        if duration_seconds < 120:  # Less than 2 minutes
            min_aligned = max(5, len(filtered_prints) * 0.005)  # 0.5% or 5 matches
        else:
            min_aligned = max(10, len(filtered_prints) * 0.01)  # 1% or 10 matches
        
        if best_episode and best_count >= min_aligned:
            # Calculate base confidence from alignment ratio
            base_confidence = (best_count / len(filtered_prints)) * 100

            # Boost confidence for strong matches (many aligned hashes)
            # This accounts for TikTok's re-encoding which reduces alignment %
            if best_count >= 100:
                # Very strong match: boost significantly
                confidence = min(95, int(base_confidence * 3))
            elif best_count >= 50:
                # Strong match: boost moderately
                confidence = min(85, int(base_confidence * 2.5))
            elif best_count >= 20:
                # Moderate match: boost slightly
                confidence = min(75, int(base_confidence * 2))
            else:
                # Weak match: use base confidence
                confidence = min(65, int(base_confidence * 1.5))

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
        """Return combined statistics from all databases."""
        all_fingerprints = self._get_all_fingerprints()
        all_counts = {}
        
        # Combine episode counts from all databases
        for db_file in self._get_all_db_files():
            db_data = self._load_db_file(db_file)
            all_counts.update(db_data['counts'])
        
        total_entries = sum(len(v) for v in all_fingerprints.values())
        return {
            "unique_hashes": len(all_fingerprints),
            "total_entries": total_entries,
            "episodes": len(all_counts),
            "avg_hashes_per_episode": total_entries / max(1, len(all_counts)),
            "databases": len(self._get_all_db_files()),
            "loaded_databases": len(self.loaded_dbs)
        }


# Global instance
audio_matcher = AudioFingerprinter()
