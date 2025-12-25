"""
LMDB Storage Layer for Audio Fingerprints

Provides fast, memory-mapped storage for audio fingerprints with:
- xxhash64 for compact hash keys (8 bytes vs 16 for MD5)
- Binary-encoded posting lists with zstd compression
- Episode ID interning for space efficiency
- Zero-copy reads via memory mapping
"""

import os
import re
import struct
import json
import lmdb
import xxhash
from typing import List, Tuple, Dict, Any, Optional


# Episode ID interning: convert between string and integer
# "S01E05" <-> 105, "S20E14" <-> 2014
def episode_to_int(episode_id: str) -> int:
    """Convert episode ID to integer (S01E05 -> 105)."""
    match = re.match(r'[Ss](\d+)[Ee](\d+)', episode_id)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        return season * 100 + episode
    return 0  # Unknown/invalid


def int_to_episode(ep_num: int) -> str:
    """Convert integer to episode ID (105 -> S01E05)."""
    season = ep_num // 100
    episode = ep_num % 100
    return f"S{season:02d}E{episode:02d}"


# Varint encoding for efficient integer storage
def encode_varint(value: int) -> bytes:
    """Encode integer as variable-length bytes (1-5 bytes for values up to 2^32)."""
    result = bytearray()
    while value >= 128:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def decode_varint(buffer: bytes, offset: int = 0) -> Tuple[int, int]:
    """
    Decode varint from buffer at offset.
    Returns (value, bytes_consumed).
    """
    result = 0
    shift = 0
    consumed = 0

    while consumed < len(buffer) - offset:
        byte = buffer[offset + consumed]
        consumed += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7

    return result, consumed


# Posting list encoding/decoding
def encode_posting_list(entries: List[Tuple[str, float]]) -> bytes:
    """
    Encode posting list to compressed binary format.

    Format (before compression):
    - uint16: number of entries
    - For each entry:
      - uint16: episode_id (interned to integer)
      - varint: timestamp in milliseconds

    Then compress entire buffer with zstd level 3.

    Args:
        entries: List of (episode_id, timestamp_seconds) tuples

    Returns:
        Compressed binary data
    """
    try:
        import zstandard as zstd
    except ImportError:
        # Fallback: no compression if zstd not available
        print("Warning: zstandard not installed, posting lists will not be compressed")
        zstd = None

    buffer = bytearray()
    buffer.extend(struct.pack('<H', len(entries)))  # 2 bytes: count

    for ep_id, time_sec in entries:
        # Intern episode ID to small integer
        ep_num = episode_to_int(ep_id)
        time_ms = int(time_sec * 1000)  # Convert to milliseconds

        buffer.extend(struct.pack('<H', ep_num))  # 2 bytes: episode
        buffer.extend(encode_varint(time_ms))     # 1-3 bytes: timestamp

    # Compress with zstd (level 3 for speed)
    if zstd:
        compressor = zstd.ZstdCompressor(level=3)
        return compressor.compress(bytes(buffer))
    else:
        return bytes(buffer)


# Thread-local storage for decompressors
import threading
_THREAD_LOCAL = threading.local()

def get_decompressor():
    if not hasattr(_THREAD_LOCAL, 'decompressor'):
        try:
            import zstandard as zstd
            _THREAD_LOCAL.decompressor = zstd.ZstdDecompressor()
        except ImportError:
            return None
    return _THREAD_LOCAL.decompressor

def decode_posting_list(data: bytes) -> List[Tuple[str, float]]:
    """
    Decode compressed posting list efficiently.
    """
    decompressor = get_decompressor()
    if decompressor:
        try:
            buffer = decompressor.decompress(data)
        except:
            buffer = data
    else:
        buffer = data

    # Fast path: use struct.unpack_from
    count = struct.unpack_from('<H', buffer, 0)[0]
    entries = []
    offset = 2
    
    # Pre-bind functions for speed
    unpack_from = struct.unpack_from
    
    for _ in range(count):
        ep_num = unpack_from('<H', buffer, offset)[0]
        offset += 2
        
        # Manual varint decoding for speed
        val = 0
        shift = 0
        while True:
            byte = buffer[offset]
            val |= (byte & 0x7f) << shift
            offset += 1
            if not (byte & 0x80):
                break
            shift += 7
            
        entries.append((int_to_episode(ep_num), val / 1000.0))

    return entries


class LMDBFingerprintStore:
    """
    LMDB-based storage for audio fingerprints.

    Features:
    - Memory-mapped for instant startup (no loading phase)
    - xxhash64 keys (8 bytes) instead of MD5 strings (16+ bytes)
    - Compressed posting lists with zstd
    - Two sub-databases: fingerprints + metadata

    Usage:
        store = LMDBFingerprintStore('/path/to/season_01.lmdb', season=1)
        store.put_hash(hash_int, [('S01E05', 12.5), ('S01E05', 15.3)])
        entries = store.get_hash(hash_int)
        store.close()
    """

    def __init__(
        self,
        db_path: str,
        season: int,
        readonly: bool = False,
        map_size: int = 500 * 1024 * 1024  # 500 MB default
    ):
        """
        Initialize LMDB fingerprint store.

        Args:
            db_path: Path to LMDB database directory
            season: Season number (for metadata)
            readonly: Open in read-only mode (for matching)
            map_size: Maximum database size in bytes
        """
        self.db_path = db_path
        self.season = season
        self.readonly = readonly

        # Create directory if it doesn't exist
        if not readonly:
            os.makedirs(db_path, exist_ok=True)

        # Open LMDB environment
        self.env = lmdb.open(
            db_path,
            map_size=map_size,
            max_dbs=2,              # fingerprints + metadata
            readonly=readonly,
            lock=not readonly,      # Lock only if writing
            readahead=True,         # Optimize for sequential reads
            meminit=False,          # Don't zero memory (faster)
            writemap=not readonly,  # Memory-mapped writes (faster)
        )

        # Store database names - we'll open them in each transaction
        # (LMDB sub-database handles may be transaction-specific)
        self.fingerprints_db_name = b'fingerprints'
        self.metadata_db_name = b'metadata'

    def put_hash(self, hash_int: int, entries: List[Tuple[str, float]]):
        """
        Store posting list for a hash.

        Args:
            hash_int: xxhash64 integer (64-bit)
            entries: List of (episode_id, timestamp_seconds) tuples
        """
        if self.readonly:
            raise RuntimeError("Cannot write to read-only database")

        key = struct.pack('<Q', hash_int)  # 8 bytes (uint64)
        value = encode_posting_list(entries)

        # Open sub-database in the same transaction where we use it
        with self.env.begin(write=True) as txn:
            fingerprints_db = self.env.open_db(self.fingerprints_db_name, txn=txn)
            txn.put(key, value, db=fingerprints_db)

    def get_hash(self, hash_int: int) -> List[Tuple[str, float]]:
        """
        Retrieve posting list for a hash.

        Args:
            hash_int: xxhash64 integer (64-bit)

        Returns:
            List of (episode_id, timestamp_seconds) tuples
        """
        key = struct.pack('<Q', hash_int)
        
        with self.env.begin(write=False) as txn:
            fingerprints_db = self.env.open_db(self.fingerprints_db_name, txn=txn)
            val = txn.get(key, db=fingerprints_db)
            if not val:
                return []
            return decode_posting_list(val)

    def get_hashes(self, hash_ints: List[int]) -> Dict[int, List[Tuple[str, float]]]:
        """
        Retrieve posting lists for multiple hashes (batch optimization).
        Opens a single transaction for all lookups.

        Args:
            hash_ints: List of xxhash64 integers

        Returns:
            Dictionary of hash_int -> list of matches
        """
        results = {}
        # Converting all to bytes first
        keys = [(h, struct.pack('<Q', h)) for h in hash_ints]
        
        with self.env.begin(write=False) as txn:
            fingerprints_db = self.env.open_db(self.fingerprints_db_name, txn=txn)
            cursor = txn.cursor(db=fingerprints_db)
            for h, key in keys:
                val = cursor.get(key)
                if val:
                    results[h] = decode_posting_list(val)
        
        return results

    def put_metadata(self, key: str, value: dict):
        """
        Store metadata (episode counts, version, etc.).

        Args:
            key: Metadata key (e.g., 'info', 'episode_counts')
            value: Dictionary to store
        """
        if self.readonly:
            raise RuntimeError("Cannot write to read-only database")

        with self.env.begin(write=True) as txn:
            metadata_db = self.env.open_db(self.metadata_db_name, txn=txn)
            txn.put(key.encode(), json.dumps(value).encode(), db=metadata_db)

    def get_metadata(self, key: str) -> dict:
        """
        Retrieve metadata.

        Args:
            key: Metadata key

        Returns:
            Dictionary (empty if not found)
        """
        with self.env.begin() as txn:
            metadata_db = self.env.open_db(self.metadata_db_name, txn=txn)
            value = txn.get(key.encode(), db=metadata_db)
            if value is None:
                return {}
            return json.loads(value.decode())

    def get_all_hashes(self) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get all fingerprints from database.

        Returns:
            Dictionary of hash_int -> posting_list
        """
        result = {}

        # Open sub-database in the same transaction where we use it
        with self.env.begin() as txn:
            fingerprints_db = self.env.open_db(self.fingerprints_db_name, txn=txn)
            cursor = txn.cursor(db=fingerprints_db)
            for key_bytes, value_bytes in cursor:
                hash_int = struct.unpack('<Q', key_bytes)[0]
                entries = decode_posting_list(value_bytes)
                result[hash_int] = entries

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with stats (unique_hashes, total_entries, etc.)
        """
        with self.env.begin() as txn:
            # Open sub-database in the same transaction where we use it
            fingerprints_db = self.env.open_db(self.fingerprints_db_name, txn=txn)
            try:
                stats = txn.stat(db=fingerprints_db)
                unique_hashes = stats['entries']
            except Exception as e:
                # Fallback: count entries manually (pass db to cursor)
                unique_hashes = sum(1 for _ in txn.cursor(db=fingerprints_db))

        metadata = self.get_metadata('info')

        return {
            'season': self.season,
            'unique_hashes': unique_hashes,
            'total_entries': metadata.get('total_entries', 0),
            'episodes': metadata.get('episodes', {}),
            'db_size_bytes': self._get_db_size(),
        }

    def _get_db_size(self) -> int:
        """Get total size of database files on disk."""
        total = 0
        if os.path.exists(self.db_path):
            for f in os.listdir(self.db_path):
                f_path = os.path.join(self.db_path, f)
                if os.path.isfile(f_path):
                    total += os.path.getsize(f_path)
        return total

    def close(self):
        """Close database environment."""
        if self.env:
            self.env.close()
            self.env = None

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.close()
