"""

CityWok API Endpoints - Audio-Only Identification

Uses Shazam-style spectral peak fingerprinting for episode identification.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from typing import Dict, Any, Optional
import os
import shutil
import uuid
import tempfile
import subprocess
from app.core.config import settings

# Audio fingerprinting
try:
    from app.services.audio_fingerprint import audio_matcher
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    audio_matcher = None
    print("Error: Audio fingerprinting not available")

# URL downloading
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
    """Download video/audio from URL using yt-dlp."""
    if not URL_DOWNLOAD_AVAILABLE:
        raise HTTPException(status_code=400, detail="URL downloads not available. Install yt-dlp.")
    
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()
    os.remove(tmp_path)
    
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
            raise HTTPException(status_code=400, detail="Failed to download from URL")
            
        return tmp_path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")


def format_time(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    m, s = divmod(int(max(0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def convert_to_mp3(input_path: str) -> str:
    """Convert any audio/video file to MP3 for consistent processing."""
    output_path = input_path.rsplit('.', 1)[0] + '.mp3'
    
    # Check if already MP3
    if input_path.lower().endswith('.mp3'):
        return input_path
    
    print(f"🔄 Converting to MP3: {os.path.basename(input_path)}")
    
    try:
        # Use ffmpeg to convert to MP3
        # -y: overwrite output file
        # -i: input file
        # -acodec libmp3lame: use MP3 codec
        # -ar 22050: sample rate (matches audio fingerprinting)
        # -ac 1: mono channel
        # -q:a 2: high quality
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', input_path,
            '-acodec', 'libmp3lame',
            '-ar', '22050',  # Match audio fingerprinting sample rate
            '-ac', '1',  # Mono
            '-q:a', '2',  # High quality
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"⚠️  FFmpeg conversion failed: {result.stderr}")
            # Fallback: return original file (librosa can handle most formats)
            return input_path
        
        # Remove original file if it was converted
        if output_path != input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass
        
        print(f"✓ Converted to MP3: {os.path.basename(output_path)}")
        return output_path
        
    except subprocess.TimeoutExpired:
        print("⚠️  Conversion timeout - using original file")
        return input_path
    except FileNotFoundError:
        print("⚠️  FFmpeg not found - using original file (librosa will handle conversion)")
        return input_path
    except Exception as e:
        print(f"⚠️  Conversion error: {e} - using original file")
        return input_path


@router.get("/test")
async def test_endpoint():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "CityWok API is running (Audio-Only)",
        "features": {
            "audio": AUDIO_AVAILABLE,
            "url_download": URL_DOWNLOAD_AVAILABLE,
        }
    }


@router.get("/stats")
async def get_stats():
    """Get database statistics."""
    if AUDIO_AVAILABLE and audio_matcher:
        return {"audio_db": audio_matcher.get_stats()}
    return {"error": "Audio service not available"}


@router.post("/identify")
async def identify_episode(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """
    Identify episode from an audio/video clip using audio fingerprinting.
    
    Accepts:
    - File upload (video or audio)
    - URL (TikTok, YouTube, etc.)
    """
    
    if not AUDIO_AVAILABLE or not audio_matcher:
        raise HTTPException(status_code=503, detail="Audio service not available")
    
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file or URL")
    
    temp_path = None
    source_name = "unknown"
    
    try:
        # Get file (from upload or URL)
        if url:
            print(f"📥 Downloading from URL: {url}")
            temp_path = download_video_from_url(url)
            source_name = url[:50]
        else:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            filename_lower = file.filename.lower()
            video_formats = ('.mp4', '.mov', '.avi', '.mpeg', '.mpg', '.mkv', '.webm')
            audio_formats = ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma')
            
            if not filename_lower.endswith(video_formats + audio_formats):
                raise HTTPException(status_code=400, detail="Invalid file format")
            
            file_id = str(uuid.uuid4())
            temp_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{file.filename}")
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file.file.close()
            
            source_name = file.filename
        
        file_size = os.path.getsize(temp_path)
        if file_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")
        
        print(f"🎵 Processing: {source_name} ({file_size / (1024*1024):.1f} MB)")
        
        # Convert to MP3 for consistent processing
        audio_path = convert_to_mp3(temp_path)
        
        # Schedule cleanup (both original and converted if different)
        background_tasks.add_task(cleanup_file, audio_path)
        if audio_path != temp_path:
            background_tasks.add_task(cleanup_file, temp_path)
        
        # Run audio analysis on MP3
        result = audio_matcher.match_clip(audio_path)
        
        if result and result.get('episode_id'):
            return {
                "match_found": True,
                "episode": result['episode_id'],
                "timestamp": format_time(result['timestamp']),
                "confidence": result.get('confidence', 0),
                "aligned_matches": result.get('aligned_matches', 0),
                "total_matches": result.get('total_matches', 0)
            }
        else:
            return {
                "match_found": False,
                "message": "No confident match found in database"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")


@router.post("/ingest")
async def ingest_episode(
    episode_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Ingest an episode into the audio database.
    """
    if not AUDIO_AVAILABLE or not audio_matcher:
        raise HTTPException(status_code=503, detail="Audio service not available")
    
    temp_path = os.path.join(settings.UPLOAD_DIR, f"ingest_{episode_id}_{file.filename}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        print(f"🎵 Processing audio for {episode_id}...")
        
        # Convert to MP3 for consistent processing
        audio_path = convert_to_mp3(temp_path)
        
        audio_matcher.add_episode(episode_id, audio_path)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, audio_path)
        if audio_path != temp_path:
            background_tasks.add_task(cleanup_file, temp_path)
        
        return {
            "status": "success",
            "episode_id": episode_id,
            "audio": True
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/admin/clear-audio")
async def clear_audio_db():
    """Clear the audio fingerprint database (admin only)."""
    if AUDIO_AVAILABLE and audio_matcher:
        audio_matcher.clear_db()
        return {"status": "success", "message": "Audio database cleared"}
    return {"status": "error", "message": "Audio service not available"}
