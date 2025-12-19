"""
Audio Fingerprinting Service with LMDB Storage

Implements Shazam-style spectral peak landmark fingerprinting with:
- LMDB memory-mapped storage (instant startup)
- xxhash64 for compact hash keys
- In-memory audio processing (no temp files)
- Backward compatible with pickle-based system
"""

import os
import re
import struct
import numpy as np
import librosa
import xxhash
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from scipy.ndimage import maximum_filter
from app.core.config import settings
from app.storage.lmdb_store import LMDBFingerprintStore


class AudioFingerprinterLMDB:
    """
    Audio fingerprinting with LMDB storage.

    Uses xxhash64 and memory-mapped databases for production deployment.
    Supports in-memory audio processing (no temp files).
    """

    # Database configuration: season_number -> database_directory
    # For local testing: only load seasons 1-5
    MAX_SEASONS = int(os.getenv('MAX_SEASONS', '5'))  # Default to 5 for local testing
    DB_CONFIG = {
        season: f"season_{season:02d}.lmdb"
        for season in range(1, MAX_SEASONS + 1)  # Seasons 1-5 (configurable)
    }

    def __init__(self, use_lmdb: bool = True):
        """
        Initialize audio fingerprinter.

        Args:
            use_lmdb: If True, use LMDB storage. If False, fall back to pickle.
        """
        self.use_lmdb = use_lmdb

        # Storage paths
        self.data_dir = settings.DATA_DIR
        if use_lmdb:
            self.fingerprint_dir = os.path.join(self.data_dir, 'fingerprints', 'v1')
        else:
            self.fingerprint_dir = self.data_dir

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

        # LMDB database handles: season -> LMDBFingerprintStore
        self.season_dbs: Dict[int, LMDBFingerprintStore] = {}
        
        # Preloaded fingerprints cache (for fast lookups)
        # hash_int -> List[Tuple[episode_id, timestamp]]
        self.fingerprints_cache: Dict[int, List[Tuple[str, float]]] = {}
        self.cache_loaded = False

        print(f"🔧 Audio Fingerprinter initialized (LMDB mode: {use_lmdb})")
        print(f"   Data directory: {self.fingerprint_dir}")

    def initialize(self):
        """Load/open all LMDB databases (lazy, memory-mapped)."""
        if not self.use_lmdb:
            print("   ℹ️  LMDB mode disabled, skipping database loading")
            return

        print("📂 Opening LMDB databases...")

        os.makedirs(self.fingerprint_dir, exist_ok=True)

        # Get max seasons from environment or use class default
        max_seasons = int(os.getenv('MAX_SEASONS', str(AudioFingerprinterLMDB.MAX_SEASONS)))
        print(f"   Loading seasons 1-{max_seasons} (MAX_SEASONS={max_seasons})")
        
        loaded_count = 0
        for season in range(1, max_seasons + 1):
            db_dir = os.path.join(self.fingerprint_dir, f"season_{season:02d}.lmdb")

            # Only open if exists
            if os.path.exists(db_dir):
                try:
                    self.season_dbs[season] = LMDBFingerprintStore(
                        db_dir,
                        season=season,
                        readonly=True  # Read-only for matching
                    )
                    loaded_count += 1
                    print(f"   ✓ Opened season {season:02d}")
                except Exception as e:
                    print(f"   ⚠️  Failed to open season {season}: {e}")

        print(f"   ✓ Opened {loaded_count} season databases")
        
        # Preload all fingerprints into memory for fast lookups
        print("📦 Preloading fingerprints into memory...")
        self._preload_fingerprints()
        print(f"   ✓ Preloaded {len(self.fingerprints_cache):,} unique hashes")

    def _get_season_from_episode_id(self, episode_id: str) -> Optional[int]:
        """Extract season number from episode ID (e.g., 'S01E05' -> 1)."""
        match = re.search(r'[Ss](\d+)', episode_id)
        if match:
            return int(match.group(1))
        return None

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

    def _create_hashes_xxh64(self, peaks: List[Tuple[int, int]]) -> List[Tuple[int, float]]:
        """
        Create fingerprint hashes using MD5 (converted to int) to match migrated data.
        
        The migration script converts MD5 strings to integers using md5_to_int(),
        so we need to use the same format for queries to match.

        Each hash encodes:
        - Anchor frequency
        - Target frequency
        - Time delta between them

        Returns list of (hash_int, anchor_time_seconds) tuples.
        """
        import hashlib
        hashes = []

        for i, (t1, f1) in enumerate(peaks):
            # Pair with nearby peaks (fan out)
            for j in range(i + 1, min(i + self.fan_value + 1, len(peaks))):
                t2, f2 = peaks[j]
                dt = t2 - t1

                if self.min_time_delta <= dt <= self.max_time_delta:
                    # Quantize frequencies and time delta for robustness
                    f1_q = f1 // 8
                    f2_q = f2 // 8
                    dt_q = dt // 4

                    # Use MD5 format to match migrated data (same as original code)
                    # Original format uses pipe separator: f"{f1_q}|{f2_q}|{dt_q}"
                    hash_input = f"{f1_q}|{f2_q}|{dt_q}"
                    md5_hex = hashlib.md5(hash_input.encode()).hexdigest()
                    md5_str = md5_hex[:16]  # First 16 hex chars (8 bytes)
                    
                    # Convert MD5 to int the same way migration does (md5_to_int function)
                    # Convert hex string to bytes, then first 8 bytes to uint64 little-endian
                    hash_bytes = bytes.fromhex(md5_str)
                    hash_int = int.from_bytes(hash_bytes[:8], byteorder='little')

                    # Store anchor time in seconds
                    anchor_time = t1 * self.hop_length / self.sr
                    hashes.append((hash_int, anchor_time))

        return hashes

    def fingerprint_audio_array(self, y: np.ndarray, sr: int = None) -> List[Tuple[int, float]]:
        """
        Generate fingerprints from audio array (not file).

        Args:
            y: Audio time series (numpy array)
            sr: Sample rate (will resample if different from self.sr)

        Returns:
            List of (hash_int, time_offset) tuples
        """
        if sr is None:
            sr = self.sr

        # Resample if needed
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)

        duration = len(y) / self.sr

        # Find spectral peaks
        peaks = self._get_spectrogram_peaks(y)

        # Create hashes from peak pairs
        hashes = self._create_hashes_xxh64(peaks)

        return hashes

    def fingerprint_audio_file(self, file_path: str) -> List[Tuple[int, float]]:
        """
        Generate fingerprints from audio file (legacy method).

        Args:
            file_path: Path to audio/video file

        Returns:
            List of (hash_int, time_offset) tuples
        """
        print(f"🎵 Fingerprinting: {os.path.basename(file_path)}")

        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            duration = len(y) / sr
            print(f"   Duration: {duration:.1f}s")

            # Generate fingerprints
            hashes = self.fingerprint_audio_array(y, sr)
            print(f"   Generated {len(hashes)} fingerprint hashes")

            return hashes

        except Exception as e:
            print(f"Error fingerprinting audio {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def add_episode_array(
        self,
        episode_id: str,
        audio_array: np.ndarray,
        sr: int = 22050
    ):
        """
        Process episode and store fingerprints in LMDB.

        Args:
            episode_id: Episode identifier (e.g., 'S01E05')
            audio_array: Audio time series
            sr: Sample rate
        """
        if not self.use_lmdb:
            raise NotImplementedError("add_episode_array requires LMDB mode")

        # Generate fingerprints
        fingerprints = self.fingerprint_audio_array(audio_array, sr)

        if not fingerprints:
            print(f"⚠️ No fingerprints generated for {episode_id}")
            return

        # Determine season
        season = self._get_season_from_episode_id(episode_id)
        if season is None:
            raise ValueError(f"Cannot determine season from episode ID: {episode_id}")

        # Open/create season database (writable mode)
        db_dir = os.path.join(self.fingerprint_dir, f"season_{season:02d}.lmdb")
        os.makedirs(db_dir, exist_ok=True)

        # Create database if not exists
        if season not in self.season_dbs:
            self.season_dbs[season] = LMDBFingerprintStore(
                db_dir,
                season=season,
                readonly=False  # Writable
            )

        db = self.season_dbs[season]

        # Group hashes by hash_int
        hash_dict = defaultdict(list)
        for hash_int, offset in fingerprints:
            hash_dict[hash_int].append((episode_id, offset))

        # Store in LMDB (merge with existing entries)
        for hash_int, new_entries in hash_dict.items():
            existing = db.get_hash(hash_int)
            combined = existing + new_entries
            db.put_hash(hash_int, combined)

        # Update metadata
        metadata = db.get_metadata('info') or {}
        episode_counts = metadata.get('episodes', {})
        episode_counts[episode_id] = len(fingerprints)
        metadata['episodes'] = episode_counts
        metadata['total_entries'] = sum(len(db.get_hash(h)) for h in hash_dict.keys())
        db.put_metadata('info', metadata)

        print(f"✓ Added {len(fingerprints)} hashes for {episode_id} to season {season}")

    def match_audio_array(
        self,
        audio_array: np.ndarray,
        sr: int = 22050
    ) -> Dict[str, Any]:
        """
        Match audio array against all LMDB databases.

        Args:
            audio_array: Query audio time series
            sr: Sample rate

        Returns:
            Match result dictionary
        """
        if not self.use_lmdb:
            raise NotImplementedError("match_audio_array requires LMDB mode")

        print("🔍 Fingerprinting query audio...")
        query_prints = self.fingerprint_audio_array(audio_array, sr)

        if not query_prints:
            print("No fingerprints found in query.")
            return {}

        print(f"   Generated {len(query_prints)} query hashes")

        # Sample if too many hashes (before searching)
        max_query_hashes = 10000
        if len(query_prints) > max_query_hashes:
            step = len(query_prints) // max_query_hashes
            query_prints = query_prints[::step]
            print(f"📊 Sampling: Using {len(query_prints)} of {len(query_prints) * step} hashes")

        print("📂 Searching across all databases...")
        
        # Use preloaded cache for fast lookups
        all_fingerprints = self._get_all_fingerprints_lmdb()
        
        if not all_fingerprints:
            print("Audio database is empty.")
            return {}

        # Filter out overly common hashes (but be less aggressive)
        max_episodes_per_hash = 100  # Increased to allow more matches
        filtered_prints = []
        skipped_common = 0
        not_found = 0
        found_count = 0

        for h, t_query in query_prints:
            if h in all_fingerprints:
                found_count += 1
                # Only skip if hash appears in MANY episodes (likely noise)
                if len(all_fingerprints[h]) > max_episodes_per_hash:
                    skipped_common += 1
                    continue
                filtered_prints.append((h, t_query))
            else:
                not_found += 1
                
        print(f"   Hash lookup: {found_count} found, {not_found} not found in database")
        if skipped_common > 0:
            print(f"⏭️  Skipped {skipped_common} common hashes (>{max_episodes_per_hash} episodes)")

        if skipped_common > 0:
            print(f"⏭️  Skipped {skipped_common} common hashes")

        print(f"Searching {len(filtered_prints)} query hashes...")

        # Find matches with time-aligned voting (fast in-memory lookups!)
        matches: Dict[str, List[float]] = defaultdict(list)
        match_count = 0

        for h, t_query in filtered_prints:
            if h in all_fingerprints:
                for ep_id, t_db in all_fingerprints[h]:
                    offset = t_db - t_query
                    matches[ep_id].append(offset)
                    match_count += 1

        print(f"Found {match_count} raw matches across {len(matches)} episodes")

        if not matches:
            return {}

        # Time-aligned voting
        best_episode = None
        best_count = 0
        best_offset = 0.0

        sorted_episodes = sorted(matches.items(), key=lambda x: len(x[1]), reverse=True)

        for ep_id, offsets in sorted_episodes:
            if len(offsets) < 5:
                continue

            # Limit processing
            if len(offsets) > 5000:
                offsets = offsets[:5000]

            # Bin offsets (0.5s bins)
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
                    print(f"✓ Early termination: Clear winner")
                    break

        print(f"Best match: {best_episode} with {best_count} aligned hashes at {best_offset:.1f}s")

        # Require minimum aligned matches
        min_aligned = max(10, len(filtered_prints) * 0.03)

        if best_episode and best_count >= min_aligned:
            confidence = min(99, int((best_count / len(filtered_prints)) * 100))
            return {
                "episode_id": best_episode,
                "timestamp": max(0, best_offset),
                "confidence": confidence,
                "aligned_matches": best_count,
                "total_matches": match_count,
                "method": "audio_lmdb"
            }

        return {}

    def _preload_fingerprints(self):
        """Preload all fingerprints from all databases into memory."""
        import time
        start_time = time.time()
        combined = {}
        total_hashes = 0

        for season, db in self.season_dbs.items():
            try:
                print(f"   Loading season {season:02d}...", flush=True)
                season_start = time.time()
                season_fps = db.get_all_hashes()
                for hash_int, entries in season_fps.items():
                    if hash_int not in combined:
                        combined[hash_int] = []
                    combined[hash_int].extend(entries)
                total_hashes += len(season_fps)
                elapsed = time.time() - season_start
                print(f"   ✓ Season {season:02d}: {len(season_fps):,} hashes ({elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"   ⚠️  Error reading season {season}: {e}", flush=True)
                import traceback
                traceback.print_exc()

        elapsed_total = time.time() - start_time
        self.fingerprints_cache = combined
        self.cache_loaded = True
        print(f"   ✓ Preload complete: {len(combined):,} unique hashes in {elapsed_total:.1f}s", flush=True)

    def _get_all_fingerprints_lmdb(self) -> Dict[int, List[Tuple[str, float]]]:
        """Get all fingerprints (from cache if preloaded, otherwise load on-demand)."""
        if self.cache_loaded:
            return self.fingerprints_cache
        
        # Fallback: load on-demand (slower)
        combined = {}
        for season, db in self.season_dbs.items():
            try:
                season_fps = db.get_all_hashes()
                for hash_int, entries in season_fps.items():
                    if hash_int not in combined:
                        combined[hash_int] = []
                    combined[hash_int].extend(entries)
            except Exception as e:
                print(f"   ⚠️  Error reading season {season}: {e}")
        return combined

    def get_stats(self) -> Dict[str, Any]:
        """Return combined statistics from all databases."""
        if not self.use_lmdb:
            return {"error": "Stats require LMDB mode"}

        total_hashes = 0
        total_entries = 0
        total_episodes = 0
        db_count = len(self.season_dbs)

        for season, db in self.season_dbs.items():
            try:
                stats = db.get_stats()
                total_hashes += stats['unique_hashes']
                total_entries += stats['total_entries']
                total_episodes += len(stats['episodes'])
            except Exception as e:
                print(f"   ⚠️  Error getting stats for season {season}: {e}")

        return {
            "unique_hashes": total_hashes,
            "total_entries": total_entries,
            "episodes": total_episodes,
            "databases": db_count,
            "storage_mode": "lmdb"
        }

    def close(self):
        """Close all LMDB databases."""
        for db in self.season_dbs.values():
            db.close()
        self.season_dbs = {}


# Global instance - use LMDB if USE_LMDB env var is set
USE_LMDB = os.getenv('USE_LMDB', 'false').lower() in ('true', '1', 'yes')
audio_matcher_lmdb = AudioFingerprinterLMDB(use_lmdb=USE_LMDB) if USE_LMDB else None
