
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form, Request, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import os
import shutil
import uuid
import tempfile
import subprocess
import json
import asyncio
from app.core.config import settings
from app.middleware.auth import verify_api_key

try:
    from app.core.audio_utils import extract_audio_to_memory
except ImportError as e:
    extract_audio_to_memory = None

try:
    from app.services.audio_fingerprint import audio_matcher
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    audio_matcher = None

try:
    from yt_dlp import YoutubeDL
    URL_DOWNLOAD_AVAILABLE = True
except ImportError:
    URL_DOWNLOAD_AVAILABLE = False

router = APIRouter()

rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 3600


def check_rate_limit(ip: str, limit: int = RATE_LIMIT_REQUESTS) -> bool:
    now = datetime.now()
    cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    rate_limit_storage[ip] = [
        req_time for req_time in rate_limit_storage[ip] 
        if req_time > cutoff
    ]
    
    if len(rate_limit_storage[ip]) >= limit:
        return False
    
    rate_limit_storage[ip].append(now)
    return True


def cleanup_file(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            pass


def get_platform_from_url(url: str) -> str:
    url_lower = url.lower()
    if 'tiktok.com' in url_lower or 'vm.tiktok.com' in url_lower:
        return 'tiktok'
    elif 'instagram.com' in url_lower or 'instagr.am' in url_lower:
        return 'instagram'
    elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    return 'unknown'


def validate_supported_url(url: str) -> bool:
    import re
    patterns = [
        r'https?://(www\.)?(tiktok\.com|vm\.tiktok\.com)',
        r'https?://(www\.)?(instagram\.com|instagr\.am)',
        r'https?://(www\.)?(youtube\.com|youtu\.be)',
    ]
    return any(re.match(pattern, url, re.IGNORECASE) for pattern in patterns)


def download_video_from_url(url: str) -> str:
    if not URL_DOWNLOAD_AVAILABLE:
        raise HTTPException(status_code=400, detail="URL downloads not available. Install yt-dlp.")
    
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()
    os.remove(tmp_path)
    
    def progress_hook(d):
        if d['status'] == 'downloading':
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            if total > 0:
                percent = (downloaded / total) * 100
                print(f"   Download progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB)")
        elif d['status'] == 'finished':
            print(f"   ✓ Download complete")
    
    ydl_opts = {
        'outtmpl': tmp_path,
        'quiet': False,
        'no_warnings': False,
        'format': 'bestaudio/worst',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': url,
        },
        'progress_hooks': [progress_hook],
        'socket_timeout': 15,
        'noplaylist': True,
        'concurrent_fragment_downloads': 4,
        'prefer_free_formats': True,
        'postprocessors': [],
    }
    
    try:
        print(f"   Starting download (timeout: 30s)...")
        import threading
        import time

        download_complete = threading.Event()
        download_error = [None]

        def download_thread():
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                download_complete.set()
            except Exception as e:
                download_error[0] = e
                download_complete.set()

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

        if not download_complete.wait(timeout=30):
            raise HTTPException(status_code=408, detail="Download timeout after 30 seconds - URL may be slow or unavailable")
        
        if download_error[0]:
            raise download_error[0]
        
        if not os.path.exists(tmp_path):
            raise HTTPException(status_code=400, detail="Failed to download from URL")
            
        return tmp_path
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        error_msg = str(e)
        print(f"   ❌ Download error: {error_msg}")

        platform = get_platform_from_url(url)
        if platform == 'instagram':
            if "Login required" in error_msg or "Private account" in error_msg or "login" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail="This Instagram content is private or requires login. Please use a public post or reel."
                )
            elif "not available" in error_msg.lower() or "removed" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail="This Instagram content is no longer available. It may have been deleted."
                )
        elif platform == 'youtube':
            if "age" in error_msg.lower() or "restricted" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail="This YouTube video is age-restricted. Please try a different video."
                )
            elif "private" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail="This YouTube video is private and cannot be accessed."
                )
        elif platform == 'tiktok':
            if "not be comfortable for some audiences" in error_msg or "Log in for access" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail="This TikTok video is age-restricted or requires login. Please try a different video."
                )

        if "Private video" in error_msg or "This video is private" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="This video is private and cannot be accessed. Please try a different video."
            )
        elif "Video not available" in error_msg or "removed" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="This video is no longer available. It may have been deleted."
            )
        else:
            raise HTTPException(status_code=400, detail=f"Download failed: {error_msg}")


def format_time(seconds: float) -> str:
    m, s = divmod(int(max(0, seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def convert_to_mp3(input_path: str) -> str:
    output_path = input_path.rsplit('.', 1)[0] + '.mp3'
    
    if input_path.lower().endswith('.mp3'):
        return input_path
    
    try:
        cmd = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-acodec', 'libmp3lame',
            '-ar', '22050',
            '-ac', '1',
            '-q:a', '2',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"⚠️  FFmpeg conversion failed: {result.stderr}")
            return input_path
        
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
    if AUDIO_AVAILABLE and audio_matcher:
        try:
            return {"audio_db": audio_matcher.get_stats()}
        except Exception as e:
            return {"error": f"Failed to get stats: {str(e)}", "message": "Service may be loading databases"}
    return {"error": "Audio service not available"}


async def _send_progress(queue: asyncio.Queue, status: str, message: str = ""):
    await queue.put({"status": status, "message": message})


async def _identify_with_progress(
    file: Optional[UploadFile],
    url: Optional[str],
    background_tasks: BackgroundTasks,
    progress_queue: asyncio.Queue
):
    temp_path = None
    source_name = "unknown"
    
    try:
        if url:
            await _send_progress(progress_queue, "downloading", "Downloading video from URL...")
            print(f"📥 Downloading from URL: {url}")
            loop = asyncio.get_event_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                temp_path = await loop.run_in_executor(
                    executor,
                    download_video_from_url,
                    url
                )
            source_name = url[:50]
        else:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            await _send_progress(progress_queue, "uploading", "Processing uploaded file...")
            
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

        await _send_progress(progress_queue, "extracting", "Extracting audio from file...")
        
        if extract_audio_to_memory and hasattr(audio_matcher, 'match_audio_array'):
            print("   Extracting audio to memory (skipping MP3 conversion)...")
            import time
            extract_start = time.time()
            try:
                loop = asyncio.get_event_loop()
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    audio_array, sr = await loop.run_in_executor(
                        executor,
                        lambda: extract_audio_to_memory(temp_path, sr=22050)
                    )
                extract_time = time.time() - extract_start
                print(f"   ✓ Audio extracted in {extract_time:.2f}s")
                
                background_tasks.add_task(cleanup_file, temp_path)
                
                await _send_progress(progress_queue, "fingerprinting", "Generating audio fingerprints...")
                
                result = await _match_with_progress(audio_matcher, audio_array, sr, progress_queue)
            except Exception as e:
                print(f"   ⚠️  In-memory extraction failed: {e}, falling back to MP3 conversion")
                await _send_progress(progress_queue, "converting", "Converting audio format...")
                loop = asyncio.get_event_loop()
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    audio_path = await loop.run_in_executor(
                        executor,
                        convert_to_mp3,
                        temp_path
                    )
                background_tasks.add_task(cleanup_file, audio_path)
                if audio_path != temp_path:
                    background_tasks.add_task(cleanup_file, temp_path)
                
                await _send_progress(progress_queue, "fingerprinting", "Generating audio fingerprints...")
                loop = asyncio.get_event_loop()
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        audio_matcher.match_clip,
                        audio_path
                    )
        else:
            await _send_progress(progress_queue, "converting", "Converting audio format...")
            print("   Converting to MP3 for consistent fingerprinting...")
            loop = asyncio.get_event_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                audio_path = await loop.run_in_executor(
                    executor,
                    convert_to_mp3,
                    temp_path
                )
            background_tasks.add_task(cleanup_file, audio_path)
            if audio_path != temp_path:
                background_tasks.add_task(cleanup_file, temp_path)
            
            await _send_progress(progress_queue, "fingerprinting", "Generating audio fingerprints...")
            loop = asyncio.get_event_loop()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    audio_matcher.match_clip,
                    audio_path
                )
        
        await _send_progress(progress_queue, "searching", "Searching database for matches...")
        
        if result:
            response = {
                "match_found": result.get('match_found', False),
                "unsure": result.get('unsure', True),
                "candidates": result.get('candidates', [])
            }
            
            # If confident match, include episode details
            if result.get('match_found'):
                await _send_progress(progress_queue, "complete", "Match found!")
                response.update({
                    "episode": result.get('episode_id'),
                    "timestamp": format_time(result.get('timestamp', 0)),
                    "confidence": result.get('confidence', 0),
                    "aligned_matches": result.get('aligned_matches', 0),
                    "total_matches": result.get('total_matches', 0)
                })
            else:
                await _send_progress(progress_queue, "complete", "Low confidence match")
                response.update({
                    "reason": result.get('reason', 'low_confidence'),
                    "message": result.get('message', "I'm not quite sure, but this is my best guess"),
                    "quality_metrics": result.get('quality_metrics', {})
                })
            
            await progress_queue.put(response)
        else:
            await _send_progress(progress_queue, "complete", "No match found")
            await progress_queue.put({
                "match_found": False,
                "unsure": True,
                "message": "No matches found in database",
                "candidates": []
            })
        
    except HTTPException as e:
        await progress_queue.put({"error": e.detail, "status_code": e.status_code})
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await progress_queue.put({"error": f"Identification failed: {str(e)}", "status_code": 500})


async def _match_with_progress(audio_matcher, audio_array, sr, progress_queue):
    import time
    from concurrent.futures import ThreadPoolExecutor
    
    await _send_progress(progress_queue, "fingerprinting", "Analyzing video...")
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        query_prints = await loop.run_in_executor(
            executor,
            audio_matcher.fingerprint_audio_array,
            audio_array,
            sr
        )
    
    if not query_prints:
        return {}
    
    await _send_progress(progress_queue, "searching", "Searching for match...")
    
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            audio_matcher.match_audio_array,
            audio_array,
            sr
        )
    
    if result and result.get('episode_id'):
        await _send_progress(progress_queue, "matching", "Analyzing match quality...")
    
    return result


@router.post("/identify")
async def identify_episode(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(verify_api_key),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    stream: Optional[str] = Form("false")
):
    """
    Identify episode from an audio/video clip using audio fingerprinting.
    """
    
    client_ip = request.client.host if request.client else "unknown"
    
    if api_key:
        if not check_rate_limit(f"auth:{client_ip}", limit=100):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Authenticated users can make 100 requests per hour. Please try again later."
            )
    else:
        if not check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. You can make 10 requests per hour. Please try again later."
            )
    
    if not AUDIO_AVAILABLE or not audio_matcher:
        raise HTTPException(status_code=503, detail="Audio service not available")

    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file or URL")

    if url and not validate_supported_url(url):
        raise HTTPException(
            status_code=400,
            detail="Unsupported URL. Please use TikTok, Instagram, or YouTube links."
        )

    stream_enabled = stream and stream.lower() in ('true', '1', 'yes')
    
    if stream_enabled:
        async def event_generator():
            progress_queue = asyncio.Queue()
            
            task = asyncio.create_task(_identify_with_progress(file, url, background_tasks, progress_queue))
            
            try:
                while True:
                    try:
                        update = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                        
                        if "error" in update:
                            yield f"data: {json.dumps(update)}\n\n"
                            break
                        elif "match_found" in update:
                            yield f"data: {json.dumps(update)}\n\n"
                            break
                        else:
                            yield f"data: {json.dumps(update)}\n\n"
                            
                    except asyncio.TimeoutError:
                        if task.done():
                            while not progress_queue.empty():
                                update = await progress_queue.get()
                                yield f"data: {json.dumps(update)}\n\n"
                            break
                        continue
                        
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'status_code': 500})}\n\n"
            finally:
                if not task.done():
                    await task
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    temp_path = None
    source_name = "unknown"
    
    try:
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

        if extract_audio_to_memory and hasattr(audio_matcher, 'match_audio_array'):
            print("   Extracting audio to memory (skipping MP3 conversion)...")
            import time
            extract_start = time.time()
            try:
                audio_array, sr = extract_audio_to_memory(temp_path, sr=22050)
                extract_time = time.time() - extract_start
                print(f"   ✓ Audio extracted in {extract_time:.2f}s")
                
                background_tasks.add_task(cleanup_file, temp_path)
                
                result = audio_matcher.match_audio_array(audio_array, sr)
            except Exception as e:
                print(f"   ⚠️  In-memory extraction failed: {e}, falling back to MP3 conversion")
                audio_path = convert_to_mp3(temp_path)
                background_tasks.add_task(cleanup_file, audio_path)
                if audio_path != temp_path:
                    background_tasks.add_task(cleanup_file, temp_path)
                result = audio_matcher.match_clip(audio_path)
        else:
            print("   Converting to MP3 for consistent fingerprinting...")
            audio_path = convert_to_mp3(temp_path)
            background_tasks.add_task(cleanup_file, audio_path)
            if audio_path != temp_path:
                background_tasks.add_task(cleanup_file, temp_path)
            result = audio_matcher.match_clip(audio_path)
        
        if result:
            response = {
                "match_found": result.get('match_found', False),
                "unsure": result.get('unsure', True),
                "candidates": result.get('candidates', [])
            }
            
            # If confident match, include episode details
            if result.get('match_found'):
                response.update({
                    "episode": result.get('episode_id'),
                    "timestamp": format_time(result.get('timestamp', 0)),
                    "confidence": result.get('confidence', 0),
                    "aligned_matches": result.get('aligned_matches', 0),
                    "total_matches": result.get('total_matches', 0)
                })
            else:
                response.update({
                    "reason": result.get('reason', 'low_confidence'),
                    "message": result.get('message', "I'm not quite sure, but this is my best guess"),
                    "quality_metrics": result.get('quality_metrics', {})
                })
            
            return response
        else:
            return {
                "match_found": False,
                "unsure": True,
                "message": "No matches found in database",
                "candidates": []
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
    if not AUDIO_AVAILABLE or not audio_matcher:
        raise HTTPException(status_code=503, detail="Audio service not available")
    
    temp_path = os.path.join(settings.UPLOAD_DIR, f"ingest_{episode_id}_{file.filename}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        print(f"🎵 Processing audio for {episode_id}...")
        
        audio_path = convert_to_mp3(temp_path)
        
        audio_matcher.add_episode(episode_id, audio_path)
        
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
    if AUDIO_AVAILABLE and audio_matcher:
        audio_matcher.clear_db()
        return {"status": "success", "message": "Audio database cleared"}
    return {"status": "error", "message": "Audio service not available"}


@router.post("/admin/force-save-databases")
async def force_save_databases():
    if AUDIO_AVAILABLE and audio_matcher:
        audio_matcher.force_save_all()
        return {"status": "success", "message": "All databases saved"}
    return {"status": "error", "message": "Audio service not available"}
