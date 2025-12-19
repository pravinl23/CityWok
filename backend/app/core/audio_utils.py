"""
Audio Extraction Utilities with FFmpeg Piping

Extracts audio from video/audio files directly to memory using FFmpeg pipes.
No temporary files created - zero disk space usage.
"""

import subprocess
import numpy as np
from typing import Tuple, BinaryIO
from fastapi import UploadFile


def extract_audio_to_memory(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Extract audio from file directly to memory via FFmpeg pipe.

    No temporary files created. FFmpeg outputs raw PCM audio to stdout,
    which is captured and converted to numpy array.

    Args:
        file_path: Path to audio/video file
        sr: Target sample rate (default: 22050 Hz)

    Returns:
        (audio_array, sample_rate) where audio_array is float32 in range [-1.0, 1.0]

    Raises:
        Exception: If FFmpeg fails
    """
    # FFmpeg command to output raw PCM audio to stdout
    cmd = [
        'ffmpeg',
        '-i', file_path,            # Input file
        '-f', 's16le',              # Output format: signed 16-bit little-endian PCM
        '-acodec', 'pcm_s16le',     # Audio codec: PCM
        '-ar', str(sr),             # Sample rate
        '-ac', '1',                 # Mono (1 channel)
        '-loglevel', 'error',       # Suppress verbose output
        '-',                        # Output to stdout
    ]

    try:
        # Run FFmpeg and capture stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Large buffer for efficiency (100 MB)
        )

        # Read audio data from stdout
        audio_bytes, stderr = process.communicate(timeout=300)  # 5 min timeout

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            raise Exception(f"FFmpeg failed (code {process.returncode}): {error_msg}")

        # Convert raw PCM bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 in range [-1.0, 1.0] (librosa format)
        audio_array = audio_array.astype(np.float32) / 32768.0

        return audio_array, sr

    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception("FFmpeg timed out after 5 minutes")
    except Exception as e:
        raise Exception(f"Audio extraction failed: {str(e)}")


def extract_audio_from_upload(
    file: UploadFile,
    sr: int = 22050
) -> Tuple[np.ndarray, int]:
    """
    Extract audio from uploaded file without writing to disk.

    Reads upload into memory and pipes through FFmpeg.
    No temporary files created.

    Args:
        file: FastAPI UploadFile object
        sr: Target sample rate (default: 22050 Hz)

    Returns:
        (audio_array, sample_rate)

    Raises:
        Exception: If FFmpeg fails or file is too large
    """
    # Read uploaded file into memory
    # Note: For very large files (>500MB), consider streaming or temp file fallback
    try:
        file_bytes = file.file.read()
    except Exception as e:
        raise Exception(f"Failed to read uploaded file: {str(e)}")

    # Check size (safety check)
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > 500:
        raise Exception(f"File too large ({file_size_mb:.1f} MB). Maximum 500 MB.")

    # FFmpeg command to read from stdin
    cmd = [
        'ffmpeg',
        '-i', 'pipe:0',             # Read from stdin
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(sr),
        '-ac', '1',
        '-loglevel', 'error',
        '-',                        # Output to stdout
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

        # Send file bytes to stdin and get audio from stdout
        audio_bytes, stderr = process.communicate(input=file_bytes, timeout=300)

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            raise Exception(f"FFmpeg failed: {error_msg}")

        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0

        return audio_array, sr

    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception("FFmpeg timed out")
    except Exception as e:
        raise Exception(f"Audio extraction failed: {str(e)}")


def extract_audio_from_bytes(
    audio_bytes: bytes,
    sr: int = 22050
) -> Tuple[np.ndarray, int]:
    """
    Extract audio from raw bytes (e.g., downloaded from URL).

    Args:
        audio_bytes: Raw audio/video file bytes
        sr: Target sample rate

    Returns:
        (audio_array, sample_rate)
    """
    cmd = [
        'ffmpeg',
        '-i', 'pipe:0',
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(sr),
        '-ac', '1',
        '-loglevel', 'error',
        '-',
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

        audio_data, stderr = process.communicate(input=audio_bytes, timeout=300)

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            raise Exception(f"FFmpeg failed: {error_msg}")

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0

        return audio_array, sr

    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception("FFmpeg timed out")
    except Exception as e:
        raise Exception(f"Audio extraction failed: {str(e)}")


def get_audio_duration(file_path: str) -> float:
    """
    Get audio duration in seconds using FFprobe.

    Args:
        file_path: Path to audio/video file

    Returns:
        Duration in seconds

    Raises:
        Exception: If FFprobe fails
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True
        )

        duration = float(result.stdout.decode().strip())
        return duration

    except (subprocess.CalledProcessError, ValueError) as e:
        raise Exception(f"Failed to get audio duration: {str(e)}")
