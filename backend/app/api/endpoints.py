from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict, Any
import os
import shutil
import uuid
import numpy as np
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
        try:
            os.remove(path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {path}: {e}")

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working."""
    return {"status": "ok", "message": "API is working"}

@router.post("/identify")
async def identify_episode(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Identify the episode from a video clip.
    """
    try:
        # Check file extension (case-insensitive)
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        filename_lower = file.filename.lower()
        if not filename_lower.endswith(('.mp4', '.mov', '.avi', '.mpeg', '.mpg')):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error checking file: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
        
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
        
        # Verify file exists and is readable
        if not os.path.exists(temp_path):
            return {"match_found": False, "message": "Temporary file was not saved correctly"}
        
        # Extract frames
        print("Extracting frames...")
        try:
            frames_data = video_processor.extract_frames(temp_path, sampling_rate=1)
        except Exception as e:
            print(f"Error extracting frames: {e}")
            import traceback
            traceback.print_exc()
            return {"match_found": False, "message": f"Error extracting frames: {str(e)}"}
            
        if not frames_data or len(frames_data) == 0:
            return {"match_found": False, "message": "No frames extracted from video"}
        
        print(f"Extracted {len(frames_data)} frames")
            
        try:
            timestamps, images = zip(*frames_data)
        except Exception as e:
            print(f"Error unpacking frames: {e}")
            return {"match_found": False, "message": f"Error processing frames: {str(e)}"}
        
        # Compute embeddings
        print("Computing embeddings...")
        try:
            embeddings = video_processor.compute_embeddings(list(images))
            print(f"Computed {len(embeddings)} embeddings")
            
            # Validate embeddings
            if len(embeddings) == 0:
                return {"match_found": False, "message": "No embeddings computed"}
            
            if embeddings.shape[0] != len(images):
                print(f"Warning: Embedding count {embeddings.shape[0]} doesn't match image count {len(images)}")
                
            print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            import traceback
            traceback.print_exc()
            return {"match_found": False, "message": f"Error processing video: {str(e)}"}
        
        # Search in Vector DB
        # Search top 5 matches for each frame
        print("Searching vector database...")
        try:
            # Limit to reasonable number of frames to search
            max_frames_to_search = 15  # Small but reasonable number
            if len(embeddings) > max_frames_to_search:
                print(f"Limiting search to first {max_frames_to_search} frames (had {len(embeddings)})")
                embeddings = embeddings[:max_frames_to_search].copy()
                timestamps = timestamps[:max_frames_to_search]
            
            # Ensure embeddings are float32 for FAISS and properly shaped
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            # Ensure 2D array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            print(f"Searching with embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            
            # Final validation before search
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print("Error: Embeddings contain NaN or Inf values")
                return {"match_found": False, "message": "Invalid embeddings computed"}
            
            search_results = vector_db.search(embeddings, k=5)
            print(f"Found {len(search_results)} search result sets")
            
            if not search_results or len(search_results) == 0:
                return {"match_found": False, "message": "No search results returned from database"}
                
        except Exception as e:
            print(f"Error searching database: {e}")
            import traceback
            traceback.print_exc()
            return {"match_found": False, "message": f"Error searching database: {str(e)}"}
        
        # Normalize query timestamps to start from 0 (in case video doesn't start at 0)
        if len(timestamps) > 0:
            first_timestamp = timestamps[0]
            normalized_timestamps = [t - first_timestamp for t in timestamps]
        else:
            normalized_timestamps = timestamps
        
        # Aggregate results (Voting)
        episode_votes = Counter()
        episode_timestamps = {}  # episode_id -> list of (estimated_start, score) tuples
        
        for frame_idx, results in enumerate(search_results):
            query_timestamp = normalized_timestamps[frame_idx]  # Now 0-based
            for score, metadata in results:
                # Filter by score threshold - use higher threshold for better matches
                if score > 0.6:  # Increased threshold for more accurate matches
                    ep_id = metadata['episode_id']
                    ep_timestamp = metadata['timestamp']
                    
                    episode_votes[ep_id] += 1
                    
                    if ep_id not in episode_timestamps:
                        episode_timestamps[ep_id] = []
                    # Estimate where the clip starts in the episode
                    # If match is at ep_time T and query frame is at q_time t (0-based), start is T - t
                    estimated_start = ep_timestamp - query_timestamp
                    episode_timestamps[ep_id].append((estimated_start, score))  # Store with score for weighting

        if episode_votes:
            best_ep_id, votes = episode_votes.most_common(1)[0]
            # Calculate estimated start time using multiple methods
            est_times_with_scores = episode_timestamps[best_ep_id]
            
            if not est_times_with_scores:
                avg_start_time = 0
            else:
                # Method 1: Use median (more robust to outliers)
                est_times = [t for t, s in est_times_with_scores]
                est_times_sorted = sorted(est_times)
                median_start_time = est_times_sorted[len(est_times_sorted) // 2]
                
                # Method 2: Use weighted average (weight by similarity score)
                total_weight = sum(s for _, s in est_times_with_scores)
                if total_weight > 0:
                    weighted_avg = sum(t * s for t, s in est_times_with_scores) / total_weight
                else:
                    weighted_avg = median_start_time
                
                # Method 3: Use earliest match (most conservative - likely most accurate)
                earliest_start = min(est_times)
                
                # Use the best strategy based on match quality
                # Filter to only high-confidence matches (score > 0.65)
                high_conf_matches = [(t, s) for t, s in est_times_with_scores if s > 0.65]
                
                if len(high_conf_matches) >= 5:
                    # If we have many high-confidence matches, use their median
                    high_conf_times = sorted([t for t, s in high_conf_matches])
                    avg_start_time = high_conf_times[len(high_conf_times) // 2]
                elif len(high_conf_matches) >= 2:
                    # If we have a few high-confidence matches, use weighted average
                    total_weight = sum(s for _, s in high_conf_matches)
                    avg_start_time = sum(t * s for t, s in high_conf_matches) / total_weight
                else:
                    # Otherwise use earliest match from all matches
                    avg_start_time = earliest_start
                
                print(f"Timestamp calculation: {len(est_times)} matches, median={median_start_time:.1f}s, weighted={weighted_avg:.1f}s, earliest={earliest_start:.1f}s, final={avg_start_time:.1f}s")
            
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

    # 2. Audio Analysis (Disabled - too inaccurate with current simple implementation)
    # TODO: Re-enable when using a proper audio fingerprinting library (e.g., dejavu, chromaprint)
    best_audio_match = None
    # Temporarily disabled due to high false positive rate
    # if AUDIO_AVAILABLE and audio_matcher:
    #     try:
    #         audio_result = audio_matcher.match_clip(temp_path)
    #         if audio_result:
    #             best_audio_match = {
    #                 "episode_id": audio_result['episode_id'],
    #                 "estimated_timestamp": max(0, audio_result['timestamp']),
    #                 "confidence": audio_result['confidence'],
    #                 "method": "audio"
    #             }
    #     except Exception as e:
    #         print(f"Audio analysis failed: {e}")
        
    # Final Decision Logic
    # Prefer visual matching as it's more reliable for this use case
    final_match = None
    if best_visual_match and best_audio_match:
        if best_visual_match['episode_id'] == best_audio_match['episode_id']:
            # Both agree - use visual with combined confidence
            final_match = best_visual_match
            final_match['confidence'] = "High (Audio+Video)"
        else:
            # Conflict: Prefer visual matching (more reliable for video clips)
            # Only use audio if visual has very low confidence
            visual_confidence = best_visual_match.get('confidence', 0)
            audio_confidence = best_audio_match.get('confidence', 0)
            
            if visual_confidence < 5 and audio_confidence > 80:
                # Visual is very uncertain, audio is very confident - use audio
                print(f"Using audio match due to low visual confidence ({visual_confidence} vs audio {audio_confidence})")
                final_match = best_audio_match
            else:
                # Prefer visual matching (default)
                print(f"Conflict: Visual={best_visual_match['episode_id']} (conf={visual_confidence}), Audio={best_audio_match['episode_id']} (conf={audio_confidence}). Preferring visual.")
                final_match = best_visual_match
    elif best_visual_match:
        final_match = best_visual_match
    elif best_audio_match:
        final_match = best_audio_match
        
    try:
        if final_match:
            # Format timestamp
            seconds = final_match['estimated_timestamp']
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            time_str = "%d:%02d:%02d" % (h, m, s)
            
            result = {
                "match_found": True,
                "episode": final_match['episode_id'],
                "timestamp": time_str,
                "details": final_match
            }
            print(f"Returning match result: {result}")
            return result
        else:
            result = {
                "match_found": False,
                "message": "Could not identify episode."
            }
            print(f"Returning no match result: {result}")
            return result
    except Exception as e:
        print(f"Error formatting response: {e}")
        import traceback
        traceback.print_exc()
        return {
            "match_found": False,
            "message": f"Error processing result: {str(e)}"
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

