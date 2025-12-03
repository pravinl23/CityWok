from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
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
            # Accept both video and audio files
            video_formats = ('.mp4', '.mov', '.avi', '.mpeg', '.mpg', '.mkv', '.webm')
            audio_formats = ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma')
            if not filename_lower.endswith(video_formats + audio_formats):
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video or audio file.")
            
            is_audio_file = filename_lower.endswith(audio_formats)
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
    
    # Determine if this is an audio-only file
    is_audio_only = False
    if file:
        filename_lower = file.filename.lower()
        audio_formats = ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma')
        is_audio_only = filename_lower.endswith(audio_formats)
    
    # 1. Video Analysis (skip if audio-only file)
    best_visual_match = None
    if not is_audio_only:
        try:
            file_size = os.path.getsize(temp_path)
            print(f"Processing: {source_name} ({file_size / (1024*1024):.1f} MB)")
            
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
    else:
        if is_audio_only:
            print("Skipping visual analysis (audio-only file)")

    # 2. Audio Analysis
    best_audio_match = None
    if AUDIO_AVAILABLE and audio_matcher:
        try:
            print("Running audio analysis...")
            audio_result = audio_matcher.match_clip(temp_path)
            if audio_result:
                best_audio_match = {
                    "episode_id": audio_result['episode_id'],
                    "estimated_timestamp": max(0, audio_result['timestamp']),
                    "confidence": audio_result['confidence'],
                    "method": "audio"
                }
                print(f"Audio match found: {best_audio_match['episode_id']} (conf={best_audio_match['confidence']})")
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
    # Hybrid Retrieval: Combine Visual + Audio Results
    visual_conf = best_visual_match.get('confidence', 0) if best_visual_match else 0
    audio_conf = best_audio_match.get('confidence', 0) if best_audio_match else 0
    
    print(f"Hybrid Retrieval - Visual: {visual_conf}, Audio: {audio_conf}")
    
    final_match = None
    
    if best_visual_match and best_audio_match:
        # Both methods found matches
        if best_visual_match['episode_id'] == best_audio_match['episode_id']:
            # Both agree - high confidence hybrid match
            final_match = best_visual_match
            final_match['confidence'] = f"High (V:{visual_conf} + A:{audio_conf})"
            final_match['method'] = "hybrid"
            print(f"✓ Hybrid match: Both methods agree on {final_match['episode_id']}")
        else:
            # Conflict: Use weighted decision
            # Audio (chroma n-grams) is very specific and robust to visual edits
            # Visual CLIP can be confused by similar scenes or edits
            
            # Calculate hybrid score
            # Audio confidence is count of aligned hashes (20+ is strong)
            # Visual confidence is vote count (15+ is decent, 30+ is strong)
            
            if audio_conf > 30 and visual_conf < 20:
                # Strong audio, weak visual -> trust audio (likely cropped/mirrored video)
                print(f"Conflict: Using audio match ({audio_conf}) - visual may be edited")
                final_match = best_audio_match
                final_match['method'] = "hybrid (audio-prioritized)"
            elif visual_conf > 40 and audio_conf < 15:
                # Strong visual, weak audio -> trust visual
                print(f"Conflict: Using visual match ({visual_conf}) - audio may be noisy")
                final_match = best_visual_match
                final_match['method'] = "hybrid (visual-prioritized)"
            elif audio_conf > visual_conf * 1.5:
                # Audio is significantly stronger
                print(f"Conflict: Using audio match ({audio_conf} vs {visual_conf})")
                final_match = best_audio_match
                final_match['method'] = "hybrid (audio-prioritized)"
            else:
                # Visual is stronger or equal
                print(f"Conflict: Using visual match ({visual_conf} vs {audio_conf})")
                final_match = best_visual_match
                final_match['method'] = "hybrid (visual-prioritized)"
                
    elif best_audio_match:
        # Only audio matched
        if audio_conf > 20:
            print(f"Only audio match found (conf={audio_conf}), accepting.")
            final_match = best_audio_match
        else:
            print(f"Only audio match found but confidence too low ({audio_conf}).")
            
    elif best_visual_match:
        # Only visual matched
        if visual_conf > 15:
            print(f"Only visual match found (conf={visual_conf}), accepting.")
            final_match = best_visual_match
        else:
            print(f"Only visual match found but confidence too low ({visual_conf}).")
        
    try:
        if final_match:
            # Format timestamp helper
            def format_time(seconds):
                m, s = divmod(int(seconds), 60)
                h, m = divmod(m, 60)
                return "%d:%02d:%02d" % (h, m, s)

            time_str = format_time(final_match['estimated_timestamp'])
            
            # Format detailed results for both methods
            visual_details = None
            if best_visual_match:
                visual_details = {
                    "episode": best_visual_match['episode_id'],
                    "timestamp": format_time(best_visual_match['estimated_timestamp']),
                    "confidence": best_visual_match['confidence']
                }

            audio_details = None
            if best_audio_match:
                audio_details = {
                    "episode": best_audio_match['episode_id'],
                    "timestamp": format_time(best_audio_match['estimated_timestamp']),
                    "confidence": best_audio_match['confidence']
                }
            
            result = {
                "match_found": True,
                "episode": final_match['episode_id'],
                "timestamp": time_str,
                "details": final_match,
                "visual_result": visual_details,
                "audio_result": audio_details
            }
            print(f"Returning hybrid match result: {result}")
            return result
        else:
            # Return partial results even if no final match
            visual_details = None
            if best_visual_match:
                def format_time(seconds):
                    m, s = divmod(int(seconds), 60)
                    h, m = divmod(m, 60)
                    return "%d:%02d:%02d" % (h, m, s)
                visual_details = {
                    "episode": best_visual_match['episode_id'],
                    "timestamp": format_time(best_visual_match['estimated_timestamp']),
                    "confidence": best_visual_match['confidence']
                }
            
            audio_details = None
            if best_audio_match:
                def format_time(seconds):
                    m, s = divmod(int(seconds), 60)
                    h, m = divmod(m, 60)
                    return "%d:%02d:%02d" % (h, m, s)
                audio_details = {
                    "episode": best_audio_match['episode_id'],
                    "timestamp": format_time(best_audio_match['estimated_timestamp']),
                    "confidence": best_audio_match['confidence']
                }
            
            result = {
                "match_found": False,
                "message": "Could not identify episode with high confidence.",
                "visual_result": visual_details,
                "audio_result": audio_details
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
    file: UploadFile = File(...),
    audio_only: bool = Query(False, description="Skip video processing, only process audio")
):
    """
    Ingest an episode into the database (Admin/Dev tool).
    
    Args:
        episode_id: Episode identifier (e.g., "S01E01")
        file: Video file to ingest
        audio_only: If True, skip video processing (only process audio)
    """
    temp_path = os.path.join(settings.UPLOAD_DIR, f"ingest_{episode_id}_{file.filename}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 1. Video (skip if audio_only=True)
    if not audio_only:
        try:
            frames = video_processor.extract_frames(temp_path, sampling_rate=0.5) # Sparse for index
            if frames:
                timestamps, images = zip(*frames)
                embeddings = video_processor.compute_embeddings(list(images))
                
                metadata = [{"episode_id": episode_id, "timestamp": t} for t in timestamps]
                vector_db.add_embeddings(embeddings, metadata)
                print(f"Video embeddings added for {episode_id}")
        except Exception as e:
            return {"status": "error", "message": f"Video processing failed: {str(e)}"}
    else:
        print(f"Skipping video processing for {episode_id} (audio_only=True)")
        
    # 2. Audio
    if AUDIO_AVAILABLE and audio_matcher:
        try:
            audio_matcher.add_episode(episode_id, temp_path)
            print(f"Audio fingerprints added for {episode_id}")
        except Exception as e:
             print(f"Audio ingestion warning: {e}")
    else:
        print("Audio fingerprinting not available")

    background_tasks.add_task(cleanup_file, temp_path)
    
    return {"status": "success", "message": f"Ingested episode {episode_id} (audio_only={audio_only})"}

