"""
CityWok API Endpoints

Hybrid video + audio identification with:
- Visual: CLIP embeddings + FAISS vector search
- Audio: Shazam-style spectral peak fingerprinting
- Hybrid confirmation: Cross-verify both modalities
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Form
from typing import Dict, Any, Optional
import os
import shutil
import uuid
import tempfile
import numpy as np
from app.services.video_processor import video_processor
from app.services.vector_db import vector_db
from app.core.config import settings
from collections import Counter

# Audio fingerprinting
try:
    from app.services.audio_fingerprint import audio_matcher
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    audio_matcher = None
    print("Warning: Audio fingerprinting not available")

# URL downloading (optional)
try:
    from yt_dlp import YoutubeDL
    URL_DOWNLOAD_AVAILABLE = True
except ImportError:
    URL_DOWNLOAD_AVAILABLE = False
    print("Warning: yt-dlp not available - URL downloads disabled")

router = APIRouter()


def cleanup_file(path: str):
    """Remove temporary file."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {path}: {e}")


def download_video_from_url(url: str) -> str:
    """Download video from URL using yt-dlp. Returns path to temp file."""
    if not URL_DOWNLOAD_AVAILABLE:
        raise HTTPException(status_code=400, detail="URL downloads not available. Install yt-dlp.")
    
    # Create temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()
    os.remove(tmp_path)  # yt-dlp needs to create the file
    
    ydl_opts = {
        'outtmpl': tmp_path,
        'quiet': True,
        'no_warnings': True,
        'format': 'best[ext=mp4]/best',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': url,
        }
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if not os.path.exists(tmp_path):
            raise HTTPException(status_code=400, detail="Failed to download video from URL")
            
        return tmp_path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")


def format_time(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    m, s = divmod(int(max(0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def has_audio_track(file_path: str) -> bool:
    """Check if a video file has an audio track."""
    try:
        import librosa
        # Try to load just a small sample (0.5 seconds) to check for audio
        # This is much faster than loading the whole file
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=0.5, offset=0.0)
        # Check if audio is not silent (has some signal)
        if len(y) == 0:
            return False
        # Check if there's actual audio content (not just silence)
        if np.max(np.abs(y)) < 0.001:  # Very quiet threshold
            return False
        return True
    except Exception as e:
        print(f"⚠️  Could not check audio track: {e}")
        # If we can't check, assume it has audio (most videos do)
        # This way we don't skip audio analysis unnecessarily
        return True


@router.get("/test")
async def test_endpoint():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "CityWok API is running",
        "features": {
            "audio": AUDIO_AVAILABLE,
            "url_download": URL_DOWNLOAD_AVAILABLE,
        }
    }


@router.get("/stats")
async def get_stats():
    """Get database statistics."""
    stats = {
        "vector_db": vector_db.get_stats(),
    }
    if AUDIO_AVAILABLE and audio_matcher:
        stats["audio_db"] = audio_matcher.get_stats()
    return stats


@router.post("/identify")
async def identify_episode(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    audio_only: bool = Form(False)
):
    """
    Identify episode from a video/audio clip.
    
    Supports:
    - File upload (video or audio)
    - URL (TikTok, YouTube, etc. via yt-dlp)
    - audio_only: If True, only use audio matching
    
    Returns hybrid match combining visual and audio analysis.
    """
    
    # Validate input
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file or URL")
    
    temp_path = None
    source_name = "unknown"
    
    try:
        # Get video file (from upload or URL)
        if url:
            print(f"📥 Downloading from URL: {url}")
            temp_path = download_video_from_url(url)
            source_name = url[:50]
        else:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            filename_lower = file.filename.lower()
            video_formats = ('.mp4', '.mov', '.avi', '.mpeg', '.mpg', '.mkv', '.webm')
            audio_formats = ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma')
            
            if not filename_lower.endswith(video_formats + audio_formats):
                raise HTTPException(status_code=400, detail="Invalid file format")
            
            # Save uploaded file
            file_id = str(uuid.uuid4())
            temp_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{file.filename}")
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file.file.close()
            
            source_name = file.filename
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")
        
        print(f"📹 Processing: {source_name} ({file_size / (1024*1024):.1f} MB)")
        
        # Determine if audio-only file (by extension)
        is_audio_file = source_name.lower().endswith(('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma'))
        
        # If user explicitly requested audio_only, or it's an audio file, skip visual
        use_audio_only = audio_only or is_audio_file
        
        # Check if video file has audio track (only for video files)
        has_audio = True
        if not is_audio_file:
            has_audio = has_audio_track(temp_path)
            if not has_audio:
                print("⚠️  No audio track detected in video file")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_path)
        
        # ===== VISUAL ANALYSIS =====
        best_visual_match = None
        if not use_audio_only:
            best_visual_match = await run_visual_analysis(temp_path)
        
        # ===== AUDIO ANALYSIS =====
        best_audio_match = None
        if AUDIO_AVAILABLE and audio_matcher and has_audio:
            # If audio_only mode: always run audio
            # Otherwise: only run audio if visual confidence < 50
            should_run_audio = use_audio_only or (best_visual_match is None or best_visual_match.get('confidence', 0) < 50)
            
            if should_run_audio:
                best_audio_match = await run_audio_analysis(temp_path)
        elif not has_audio:
            print("⏭️  Skipping audio analysis (no audio track)")
        
        # ===== HYBRID CONFIRMATION =====
        return create_hybrid_result(best_visual_match, best_audio_match)
        
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in identify_episode: {error_msg}")
        import traceback
        traceback.print_exc()
        # Return more detailed error message
        raise HTTPException(
            status_code=500, 
            detail=f"Identification failed: {error_msg}. Check server logs for details."
        )


async def run_visual_analysis(temp_path: str) -> Optional[Dict[str, Any]]:
    """Run CLIP-based visual analysis on video."""
    try:
        print("🎬 Running visual analysis...")
        
        # Extract keyframes (scene change detection)
        frames_data = video_processor.extract_keyframes(temp_path, max_frames=50)
        
        if not frames_data:
            print("No frames extracted")
            return None
        
        timestamps, images = zip(*frames_data)
        
        # Compute CLIP embeddings
        embeddings = video_processor.compute_embeddings(list(images))
        
        if len(embeddings) == 0:
            print("No embeddings computed")
            return None
        
        # Limit search to prevent timeouts
        max_search = min(30, len(embeddings))
        if len(embeddings) > max_search:
            embeddings = embeddings[:max_search]
            timestamps = timestamps[:max_search]
        
        # Normalize timestamps to start from 0
        first_ts = timestamps[0]
        norm_timestamps = [t - first_ts for t in timestamps]
        
        # Search vector database
        print(f"🔍 Searching {len(embeddings)} embeddings...")
        search_results = vector_db.search(embeddings, k=5)
        
        if not search_results or len(search_results) == 0:
            print("No search results returned")
            return None
        
        # Aggregate votes
        episode_votes = Counter()
        episode_timestamps = {}
        
        for frame_idx, results in enumerate(search_results):
            if not results:
                continue
            query_ts = norm_timestamps[frame_idx]
            for score, metadata in results:
                if not isinstance(metadata, dict):
                    print(f"Warning: Invalid metadata format: {type(metadata)}")
                    continue
                if score > 0.55:  # Similarity threshold
                    ep_id = metadata.get('episode_id')
                    if not ep_id:
                        continue
                    ep_ts = metadata.get('timestamp', 0)
                    
                    episode_votes[ep_id] += 1
                    
                    if ep_id not in episode_timestamps:
                        episode_timestamps[ep_id] = []
                    
                    # Estimate clip start in episode
                    estimated_start = ep_ts - query_ts
                    episode_timestamps[ep_id].append((estimated_start, score))
        
        if not episode_votes:
            print("No episode matches found")
            return None
        
        # Get best match
        best_ep_id, votes = episode_votes.most_common(1)[0]
        est_times = episode_timestamps[best_ep_id]
        
        # Calculate timestamp using weighted median
        if est_times:
            sorted_times = sorted(est_times, key=lambda x: x[1], reverse=True)
            high_conf = [t for t, s in sorted_times if s > 0.6]
            
            if len(high_conf) >= 3:
                final_ts = sorted(high_conf)[len(high_conf) // 2]
            else:
                final_ts = min(t for t, s in est_times)
        else:
            final_ts = 0
        
        print(f"✓ Visual match: {best_ep_id} @ {format_time(final_ts)} ({votes} votes)")
        
        return {
            "episode_id": best_ep_id,
            "estimated_timestamp": max(0, final_ts),
            "confidence": votes,
            "method": "visual"
        }
        
    except Exception as e:
        print(f"Visual analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_audio_analysis(temp_path: str) -> Optional[Dict[str, Any]]:
    """Run spectral peak fingerprint analysis on audio."""
    try:
        print("🎵 Running audio analysis...")
        
        result = audio_matcher.match_clip(temp_path)
        
        if result:
            print(f"✓ Audio match: {result['episode_id']} @ {format_time(result['timestamp'])} "
                  f"({result.get('aligned_matches', result['confidence'])} aligned)")
            return {
                "episode_id": result['episode_id'],
                "estimated_timestamp": max(0, result['timestamp']),
                "confidence": result.get('aligned_matches', result['confidence']),
                "method": "audio"
            }
        
        print("No audio match found")
        return None
        
    except Exception as e:
        print(f"Audio analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_hybrid_result(
    visual: Optional[Dict[str, Any]],
    audio: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combine visual and audio results with hybrid confirmation.
    
    Decision logic:
    1. If both agree -> High confidence
    2. If conflict -> Use weighted scoring
    3. If only one -> Accept if above threshold
    """
    
    # Safely extract confidence values
    visual_conf = 0
    if visual and isinstance(visual, dict):
        visual_conf = visual.get('confidence', 0)
        if not isinstance(visual_conf, (int, float)):
            visual_conf = 0
    
    audio_conf = 0
    if audio and isinstance(audio, dict):
        audio_conf = audio.get('confidence', 0)
        if not isinstance(audio_conf, (int, float)):
            audio_conf = 0
    
    print(f"🔀 Hybrid decision: Visual={visual_conf}, Audio={audio_conf}")
    
    final_match = None
    
    if visual and audio:
        if visual['episode_id'] == audio['episode_id']:
            # Both agree - high confidence!
            final_match = visual.copy()
            final_match['method'] = "hybrid (confirmed)"
            final_match['hybrid_confidence'] = f"V:{visual_conf}+A:{audio_conf}"
            print(f"✓ CONFIRMED: Both methods agree on {final_match['episode_id']}")
            
            # Use timestamp from higher confidence method
            if audio_conf > visual_conf * 2:
                final_match['estimated_timestamp'] = audio['estimated_timestamp']
        else:
            # Conflict - weighted decision
            # Audio fingerprints are very specific (hash-based)
            # Visual can be confused by similar scenes
            
            if audio_conf >= 30 and visual_conf < 15:
                print(f"⚠️ Conflict: Trusting audio ({audio_conf} >> {visual_conf})")
                final_match = audio.copy()
                final_match['method'] = "hybrid (audio wins)"
            elif visual_conf >= 30 and audio_conf < 15:
                print(f"⚠️ Conflict: Trusting visual ({visual_conf} >> {audio_conf})")
                final_match = visual.copy()
                final_match['method'] = "hybrid (visual wins)"
            elif audio_conf > visual_conf * 1.5:
                print(f"⚠️ Conflict: Audio stronger ({audio_conf} > {visual_conf})")
                final_match = audio.copy()
                final_match['method'] = "hybrid (audio stronger)"
            else:
                print(f"⚠️ Conflict: Visual stronger ({visual_conf} >= {audio_conf})")
                final_match = visual.copy()
                final_match['method'] = "hybrid (visual stronger)"
    
    elif audio and audio_conf >= 15:
        print(f"Audio-only match (conf={audio_conf})")
        final_match = audio
        
    elif visual and visual_conf >= 10:
        print(f"Visual-only match (conf={visual_conf})")
        final_match = visual
    
    # Format response
    visual_result = None
    if visual:
        visual_result = {
            "episode": visual['episode_id'],
            "timestamp": format_time(visual['estimated_timestamp']),
            "confidence": visual['confidence']
        }
    
    audio_result = None
    if audio:
        audio_result = {
            "episode": audio['episode_id'],
            "timestamp": format_time(audio['estimated_timestamp']),
            "confidence": audio['confidence']
        }
    
    if final_match:
        return {
            "match_found": True,
            "episode": final_match['episode_id'],
            "timestamp": format_time(final_match['estimated_timestamp']),
            "confidence": final_match['confidence'],
            "method": final_match['method'],
            "visual_result": visual_result,
            "audio_result": audio_result
        }
    else:
        return {
            "match_found": False,
            "message": "No confident match found",
            "visual_result": visual_result,
            "audio_result": audio_result
        }


@router.post("/admin/clear-audio")
async def clear_audio_db():
    """Clear the audio fingerprint database (admin only)."""
    if AUDIO_AVAILABLE and audio_matcher:
        audio_matcher.clear_db()
        return {"status": "success", "message": "Audio database cleared"}
    return {"status": "error", "message": "Audio service not available"}


@router.post("/ingest")
async def ingest_episode(
    episode_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    audio_only: bool = Query(False, description="Skip video, only process audio")
):
    """
    Ingest an episode into the database.
    
    Processes both video (CLIP embeddings) and audio (spectral fingerprints).
    """
    temp_path = os.path.join(settings.UPLOAD_DIR, f"ingest_{episode_id}_{file.filename}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = {"episode_id": episode_id, "video": False, "audio": False}
    
    # Video processing
    if not audio_only:
        try:
            print(f"📹 Processing video for {episode_id}...")
            
            # Use keyframe extraction for efficiency
            frames = video_processor.extract_keyframes(temp_path, max_frames=200)
            
            if frames:
                timestamps, images = zip(*frames)
                embeddings = video_processor.compute_embeddings(list(images))
                
                metadata = [{"episode_id": episode_id, "timestamp": t} for t in timestamps]
                vector_db.add_embeddings(embeddings, metadata)
                
                results["video"] = True
                results["video_frames"] = len(frames)
                print(f"✓ Added {len(frames)} video embeddings for {episode_id}")
        except Exception as e:
            results["video_error"] = str(e)
            print(f"Video error: {e}")
    
    # Audio processing
    if AUDIO_AVAILABLE and audio_matcher:
        try:
            print(f"🎵 Processing audio for {episode_id}...")
            audio_matcher.add_episode(episode_id, temp_path)
            results["audio"] = True
        except Exception as e:
            results["audio_error"] = str(e)
            print(f"Audio error: {e}")
    
    background_tasks.add_task(cleanup_file, temp_path)
    
    return {"status": "success", "results": results}
