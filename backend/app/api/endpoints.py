from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict, Any
import os
import shutil
import uuid
from app.services.video_processor import video_processor
from app.services.vector_db import vector_db

# Audio fingerprinting is optional
try:
    from app.services.audio_fingerprint import audio_matcher
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    audio_matcher = None
    print("Warning: Audio fingerprinting not available")
from app.core.config import settings
from collections import Counter

router = APIRouter()

def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)

@router.post("/identify")
async def identify_episode(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Identify the episode from a video clip.
    """
    # Check file extension (case-insensitive)
    filename_lower = file.filename.lower() if file.filename else ""
    if not filename_lower.endswith(('.mp4', '.mov', '.avi', '.mpeg', '.mpg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
        
    # Save temp file
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{file.filename}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size after saving
        file_size = os.path.getsize(temp_path)
        
        # Check file size (max 100MB)
        if file_size > 100 * 1024 * 1024:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 100MB.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    finally:
        file.file.close()
        
    # Schedule cleanup
    background_tasks.add_task(cleanup_file, temp_path)
    
    # 1. Video Analysis
    best_visual_match = None
    try:
        print(f"Processing file: {file.filename} ({file_size / (1024*1024):.1f} MB)")
        # Extract frames
        print("Extracting frames...")
        frames_data = video_processor.extract_frames(temp_path, sampling_rate=1)
        if not frames_data:
             return {"match_found": False, "message": "No frames extracted from video"}
        
        print(f"Extracted {len(frames_data)} frames")
        if len(frames_data) == 0:
            return {"match_found": False, "message": "No frames extracted from video"}
            
        timestamps, images = zip(*frames_data)
        
        # Compute embeddings
        print("Computing embeddings...")
        try:
            embeddings = video_processor.compute_embeddings(list(images))
            print(f"Computed {len(embeddings)} embeddings")
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            import traceback
            traceback.print_exc()
            return {"match_found": False, "message": f"Error processing video: {str(e)}"}
        
        # Search in Vector DB
        # Search top 5 matches for each frame
        print("Searching vector database...")
        try:
            search_results = vector_db.search(embeddings, k=5)
            print(f"Found {len(search_results)} search result sets")
        except Exception as e:
            print(f"Error searching database: {e}")
            import traceback
            traceback.print_exc()
            return {"match_found": False, "message": f"Error searching database: {str(e)}"}
        
        # Aggregate results (Voting)
        episode_votes = Counter()
        episode_timestamps = {}  # episode_id -> list of timestamps
        
        for frame_idx, results in enumerate(search_results):
            query_timestamp = timestamps[frame_idx]
            for score, metadata in results:
                # Filter by score threshold (e.g., 0.8) if needed
                if score > 0.5: # Lower threshold for MVP
                    ep_id = metadata['episode_id']
                    ep_timestamp = metadata['timestamp']
                    
                    episode_votes[ep_id] += 1
                    
                    if ep_id not in episode_timestamps:
                        episode_timestamps[ep_id] = []
                    # Estimate where the clip starts in the episode
                    # If match is at ep_time T and query is at q_time t, start is T - t
                    episode_timestamps[ep_id].append(ep_timestamp - query_timestamp)

        if episode_votes:
            best_ep_id, votes = episode_votes.most_common(1)[0]
            # Calculate average estimated start time
            est_times = episode_timestamps[best_ep_id]
            avg_start_time = sum(est_times) / len(est_times)
            
            best_visual_match = {
                "episode_id": best_ep_id,
                "estimated_timestamp": max(0, avg_start_time),
                "confidence": votes,
                "method": "visual"
            }
            print(f"Best visual match: {best_ep_id} with {votes} votes")
        else:
            print("No episode votes found")
            
    except Exception as e:
        import traceback
        print(f"Visual analysis failed: {e}")
        traceback.print_exc()
        best_visual_match = None

    # 2. Audio Analysis (Fallback or Confirmation)
    best_audio_match = None
    if AUDIO_AVAILABLE and audio_matcher:
        try:
            audio_result = audio_matcher.match_clip(temp_path)
            if audio_result:
                best_audio_match = {
                    "episode_id": audio_result['episode_id'],
                    "estimated_timestamp": max(0, audio_result['timestamp']),
                    "confidence": audio_result['confidence'],
                    "method": "audio"
                }
        except Exception as e:
            print(f"Audio analysis failed: {e}")
        
    # Final Decision Logic
    final_match = None
    if best_visual_match and best_audio_match:
        if best_visual_match['episode_id'] == best_audio_match['episode_id']:
            final_match = best_visual_match
            final_match['confidence'] = "High (Audio+Video)"
        else:
            # Conflict: Prefer audio for exactness if confidence is high, else video
            # For now, return both or prefer video if votes are high
            final_match = best_audio_match # Prefer audio for now as it's usually exact
    elif best_visual_match:
        final_match = best_visual_match
    elif best_audio_match:
        final_match = best_audio_match
        
    if final_match:
        # Format timestamp
        seconds = final_match['estimated_timestamp']
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        time_str = "%d:%02d:%02d" % (h, m, s)
        
        return {
            "match_found": True,
            "episode": final_match['episode_id'],
            "timestamp": time_str,
            "details": final_match
        }
    else:
        return {
            "match_found": False,
            "message": "Could not identify episode."
        }

@router.post("/ingest")
async def ingest_episode(
    episode_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Ingest an episode into the database (Admin/Dev tool).
    """
    temp_path = os.path.join(settings.UPLOAD_DIR, f"ingest_{episode_id}_{file.filename}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process immediately for MVP simplicity (should be background task)
    # 1. Video
    try:
        frames = video_processor.extract_frames(temp_path, sampling_rate=0.5) # Sparse for index
        if frames:
            timestamps, images = zip(*frames)
            embeddings = video_processor.compute_embeddings(list(images))
            
            metadata = [{"episode_id": episode_id, "timestamp": t} for t in timestamps]
            vector_db.add_embeddings(embeddings, metadata)
    except Exception as e:
        return {"status": "error", "message": f"Video processing failed: {str(e)}"}
        
    # 2. Audio
    if AUDIO_AVAILABLE and audio_matcher:
        try:
            audio_matcher.add_episode(episode_id, temp_path)
        except Exception as e:
             print(f"Audio ingestion warning: {e}")

    background_tasks.add_task(cleanup_file, temp_path)
    
    return {"status": "success", "message": f"Ingested episode {episode_id}"}

