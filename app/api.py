"""
FastAPI Backend for VideoEmotion Administration System.
Provides REST API endpoints for video management, trash operations, and pipeline execution.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pathlib import Path
from datetime import datetime
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core import (
    VideoManager,
    TrashManager,
    StatsUpdater,
    PipelineExecutor,
    VideoMode,
    VideoStatus,
    PipelineConfig,
)
from app.dependencies import (
    get_video_manager,
    get_trash_manager,
    get_stats_updater,
    get_pipeline_executor,
)

from fastapi.staticfiles import StaticFiles

# ... (existing imports)

# Initialize FastAPI app
app = FastAPI(
    title="VideoEmotion API",
    description="REST API for video emotion analysis administration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static/videos", StaticFiles(directory=project_root / "data" / "videos"), name="videos")
app.mount("/static/output", StaticFiles(directory=project_root / "output"), name="output")

# Pydantic models for API
class VideoResponse(BaseModel):
    id: str
    name: str
    mode: str
    status: str
    created_at: str
    processed_at: Optional[str]
    file_size_mb: Optional[float]
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    stats: Optional[dict] = None

class VideoListResponse(BaseModel):
    videos: List[VideoResponse]
    total: int
    pagination: dict

class TrashItemResponse(BaseModel):
    trash_id: str
    video_name: str
    mode: str
    original_status: str
    deleted_at: str
    size_mb: Optional[float]

class PipelineRunRequest(BaseModel):
    video_name: str
    options: Optional[dict] = None

class PipelineJobResponse(BaseModel):
    job_id: str
    video_name: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: Optional[dict]
    logs: List[str]
    error: Optional[str]

class StatsResponse(BaseModel):
    total_videos: int
    offline_videos: int
    realtime_videos: int
    processed: int
    partial: int
    unprocessed: int
    total_size_mb: float
    trash_stats: dict
    emotion_distribution: Optional[dict] = None


# ============================================================================
# Video Management Endpoints
# ============================================================================

@app.get("/api/videos", response_model=VideoListResponse)
async def list_videos(
    mode: Optional[str] = Query(None, description="Filter by mode: offline, realtime"),
    status: Optional[str] = Query(None, description="Filter by status: processed, partial, unprocessed"),
    sort_by: str = Query("created_at", description="Sort by: name, created_at, status"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    video_manager: VideoManager = Depends(get_video_manager)
):
    """List all active videos with optional filters"""
    try:
        video_mode = VideoMode(mode) if mode else None
        video_status = VideoStatus(status) if status else None
        
        videos = video_manager.list_videos(
            mode=video_mode,
            status=video_status,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Pagination
        total = len(videos)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_videos = videos[start_idx:end_idx]
        
        # Convert to response format
        video_responses = []
        for v in paginated_videos:
            # Determine video URL
            viz_h264 = project_root / "output" / "visualizations" / v.name / f"{v.name}_annotated_h264.mp4"
            viz_raw = project_root / "output" / "visualizations" / v.name / f"{v.name}_annotated_raw.mp4"
            realtime_h264 = project_root / "output" / "realtime" / v.name / "session_h264.mp4"
            
            if viz_h264.exists():
                video_url = f"/static/output/visualizations/{v.name}/{v.name}_annotated_h264.mp4"
            elif viz_raw.exists():
                 video_url = f"/static/output/visualizations/{v.name}/{v.name}_annotated_raw.mp4"
            elif realtime_h264.exists():
                 video_url = f"/static/output/realtime/{v.name}/session_h264.mp4"
            else:
                video_url = f"/static/videos/{v.name}/{v.name}.mp4"
            
            # Determine thumbnail URL
            thumbnail_url = f"/static/output/thumbnails/{v.name}.jpg"

            video_responses.append(VideoResponse(
                id=v.id,
                name=v.name,
                mode=v.mode.value,
                status=v.status.value,
                created_at=v.created_at.isoformat(),
                processed_at=v.processed_at.isoformat() if v.processed_at else None,
                file_size_mb=round(v.file_size_bytes / (1024 * 1024), 2) if v.file_size_bytes else None,
                video_url=video_url,
                thumbnail_url=thumbnail_url,
                stats=v.stats
            ))
        
        return VideoListResponse(
            videos=video_responses,
            total=total,
            pagination={
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to list videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/videos/unprocessed")
async def list_unprocessed_videos():
    """List unprocessed offline videos"""
    try:
        videos = video_manager.get_unprocessed_videos()
        
        video_responses = []
        for v in videos:
            video_responses.append(VideoResponse(
                id=v.id,
                name=v.name,
                mode=v.mode.value,
                status=v.status.value,
                created_at=v.created_at.isoformat(),
                processed_at=None,
                file_size_mb=round(v.file_size_bytes / (1024 * 1024), 2) if v.file_size_bytes else None,
                stats=None
            ))
        
        return {"videos": video_responses, "total": len(video_responses)}
    
    except Exception as e:
        logger.error(f"Failed to list unprocessed videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    video_manager: VideoManager = Depends(get_video_manager)
):
    """Get video details by ID"""
    video = video_manager.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Determine video URL
    viz_h264 = project_root / "output" / "visualizations" / video.name / f"{video.name}_annotated_h264.mp4"
    viz_raw = project_root / "output" / "visualizations" / video.name / f"{video.name}_annotated_raw.mp4"
    realtime_h264 = project_root / "output" / "realtime" / video.name / "session_h264.mp4"
    
    if viz_h264.exists():
        video_url = f"/static/output/visualizations/{video.name}/{video.name}_annotated_h264.mp4"
    elif viz_raw.exists():
        video_url = f"/static/output/visualizations/{video.name}/{video.name}_annotated_raw.mp4"
    elif realtime_h264.exists():
        video_url = f"/static/output/realtime/{video.name}/session_h264.mp4"
    else:
        video_url = f"/static/videos/{video.name}/{video.name}.mp4"
    
    # Determine thumbnail URL
    thumbnail_url = f"/static/output/thumbnails/{video.name}.jpg"

    return VideoResponse(
        id=video.id,
        name=video.name,
        mode=video.mode.value,
        status=video.status.value,
        created_at=video.created_at.isoformat(),
        processed_at=video.processed_at.isoformat() if video.processed_at else None,
        file_size_mb=round(video.file_size_bytes / (1024 * 1024), 2) if video.file_size_bytes else None,
        video_url=video_url,
        thumbnail_url=thumbnail_url,
        stats=video.stats
    )


@app.post("/api/videos/scan")
async def scan_videos(
    background_tasks: BackgroundTasks,
    video_manager: VideoManager = Depends(get_video_manager)
):
    """Trigger video scan to update inventory"""
    try:
        background_tasks.add_task(video_manager.scan_videos_async)
        return {"message": "Video scan started", "status": "pending"}
    except Exception as e:
        logger.error(f"Failed to start video scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.delete("/api/videos/{video_id}")
async def delete_video(
    video_id: str, 
    background_tasks: BackgroundTasks,
    video_manager: VideoManager = Depends(get_video_manager),
    trash_manager: TrashManager = Depends(get_trash_manager),
    stats_updater: StatsUpdater = Depends(get_stats_updater)
):
    """Move video to trash"""
    video = video_manager.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Move to trash
        trash_meta = trash_manager.move_to_trash(video)
        
        # Update video metadata
        video_manager.scan_videos()
        
        # Recalculate stats in background
        background_tasks.add_task(stats_updater.recalculate_all_stats)
        
        logger.info(f"Deleted video: {video_id}")
        
        return {
            "success": True,
            "message": "Video moved to trash",
            "trash_id": trash_meta.trash_id,
            "stats_updated": "pending"
        }
    
    except Exception as e:
        logger.error(f"Failed to delete video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Trash Management Endpoints
# ============================================================================

@app.get("/api/trash")
async def list_trash(trash_manager: TrashManager = Depends(get_trash_manager)):
    """List all items in trash"""
    try:
        trash_items = trash_manager.list_trash()
        
        responses = []
        for item in trash_items:
            responses.append(TrashItemResponse(
                trash_id=item.trash_id,
                video_name=item.video_name,
                mode=item.mode.value,
                original_status=item.original_status.value,
                deleted_at=item.deleted_at.isoformat(),
                size_mb=round(item.size_bytes / (1024 * 1024), 2) if item.size_bytes else None
            ))
        
        return {"trash_items": responses, "total": len(responses)}
    
    except Exception as e:
        logger.error(f"Failed to list trash: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trash/{trash_id}/restore")
async def restore_video(
    trash_id: str, 
    background_tasks: BackgroundTasks,
    trash_manager: TrashManager = Depends(get_trash_manager),
    video_manager: VideoManager = Depends(get_video_manager),
    stats_updater: StatsUpdater = Depends(get_stats_updater)
):
    """Restore video from trash"""
    try:
        success, original_video_id = trash_manager.restore_from_trash(trash_id)
        
        if success:
            # Re-scan to update metadata
            background_tasks.add_task(video_manager.scan_videos_async)
            background_tasks.add_task(stats_updater.recalculate_all_stats)
            
            logger.info(f"Restored video: {trash_id}")
            
            return {
                "success": True,
                "message": "Video restored successfully",
                "video_id": original_video_id
            }
        else:
            raise HTTPException(status_code=500, detail="Restoration failed")
    
    except Exception as e:
        logger.error(f"Failed to restore {trash_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/trash/{trash_id}")
async def delete_permanently(
    trash_id: str,
    trash_manager: TrashManager = Depends(get_trash_manager)
):
    """Permanently delete from trash"""
    try:
        size_bytes = trash_manager.delete_permanently(trash_id)
        
        logger.info(f"Permanently deleted: {trash_id}")
        
        return {
            "success": True,
            "message": "Video permanently deleted",
            "freed_space_mb": round(size_bytes / (1024 * 1024), 2)
        }
    
    except Exception as e:
        logger.error(f"Failed to permanently delete {trash_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trash/empty")
async def empty_trash(
    trash_manager: TrashManager = Depends(get_trash_manager)
):
    """Empty entire trash"""
    try:
        count, total_size = trash_manager.empty_trash()
        
        logger.info(f"Emptied trash: {count} items")
        
        return {
            "success": True,
            "message": f"Deleted {count} items",
            "freed_space_mb": round(total_size / (1024 * 1024), 2)
        }
    
    except Exception as e:
        logger.error(f"Failed to empty trash: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Pipeline Execution Endpoints
# ============================================================================

@app.post("/api/pipeline/run", response_model=PipelineJobResponse)
async def run_pipeline(
    request: PipelineRunRequest, 
    background_tasks: BackgroundTasks,
    pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor),
    video_manager: VideoManager = Depends(get_video_manager)
):
    """Start pipeline execution for a video"""
    try:
        # Create config from options
        options = request.options or {}
        config = PipelineConfig(
            fps=options.get("fps", 5),
            smoothing=options.get("smoothing", True),
            tta=options.get("tta", True),
            backend=options.get("backend", "hsemotion"),
            visualize=options.get("visualize", True),
            overwrite=options.get("overwrite", False),
            export_bboxes=options.get("export_bboxes", True),
            no_extract=options.get("no_extract", False),
            no_detect=options.get("no_detect", False),
            no_analyze=options.get("no_analyze", False),
            no_summary=options.get("no_summary", False),
            no_visualize=options.get("no_visualize", False),
        )
        
        # Create job
        job_id = pipeline_executor.create_job(request.video_name, config)
        
        # Define wrapper to update video manager after execution
        def run_pipeline_and_refresh(jid: str):
            pipeline_executor.execute_job(jid)
            # Re-scan to detect new output files and update status
            video_manager.scan_videos()
            logger.info(f"Triggered video scan after pipeline job {jid}")

        # Start execution in background
        background_tasks.add_task(run_pipeline_and_refresh, job_id)
        
        job = pipeline_executor.get_job(job_id)
        
        logger.info(f"Started pipeline job: {job_id} for {request.video_name}")
        
        return PipelineJobResponse(
            job_id=job.job_id,
            video_name=job.video_name,
            status=job.status.value,
            created_at=job.created_at.isoformat(),
            started_at=None,
            completed_at=None,
            progress=None,
            logs=[],
            error=None
        )
    
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipeline/jobs/{job_id}", response_model=PipelineJobResponse)
async def get_job_status(
    job_id: str,
    pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor)
):
    """Get pipeline job status"""
    job = pipeline_executor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return PipelineJobResponse(
        job_id=job.job_id,
        video_name=job.video_name,
        status=job.status.value,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress.to_dict() if job.progress else None,
        logs=job.logs[-50:],  # Last 50 log lines
        error=job.error
    )


@app.get("/api/pipeline/jobs")
async def list_jobs(
    limit: int = Query(50, ge=1, le=100),
    pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor)
):
    """List recent pipeline jobs"""
    try:
        jobs = pipeline_executor.list_jobs(limit=limit)
        
        responses = []
        for job in jobs:
            responses.append(PipelineJobResponse(
                job_id=job.job_id,
                video_name=job.video_name,
                status=job.status.value,
                created_at=job.created_at.isoformat(),
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                progress=job.progress.to_dict() if job.progress else None,
                logs=job.logs[-10:],  # Last 10 log lines for list view
                error=job.error
            ))
        
        return {"jobs": responses, "total": len(responses)}
    
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/pipeline/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor)
):
    """Cancel a running pipeline job"""
    try:
        success = pipeline_executor.cancel_job(job_id)
        
        if success:
            return {"success": True, "message": "Job cancelled"}
        else:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled (not running or not found)")
    
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics Endpoints
# ============================================================================

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(
    video_manager: VideoManager = Depends(get_video_manager),
    trash_manager: TrashManager = Depends(get_trash_manager)
):
    """Get global statistics"""
    try:
        video_stats = video_manager.get_stats()
        trash_stats = trash_manager.get_trash_stats()
        
        return StatsResponse(
            total_videos=video_stats["total_videos"],
            offline_videos=video_stats["offline_videos"],
            realtime_videos=video_stats["realtime_videos"],
            processed=video_stats["processed"],
            partial=video_stats["partial"],
            unprocessed=video_stats["unprocessed"],
            total_size_mb=video_stats["total_size_mb"],
            trash_stats=trash_stats,
            emotion_distribution=video_stats.get("emotion_distribution")
        )
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stats/refresh")
async def refresh_stats(
    background_tasks: BackgroundTasks,
    stats_updater: StatsUpdater = Depends(get_stats_updater)
):
    """Force statistics recalculation"""
    try:
        background_tasks.add_task(stats_updater.recalculate_all_stats)
        return {"message": "Statistics refresh started", "status": "pending"}
    except Exception as e:
        logger.error(f"Failed to refresh stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "VideoEmotion API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("VideoEmotion API starting up...")
    
    # Manually resolve dependencies for startup
    # Since we removed global variables
    # Note: get_video_manager uses lru_cache so it returns the singleton
    video_manager = get_video_manager()
    pipeline_executor = get_pipeline_executor()
    
    # Scan videos on startup
    try:
        video_manager.scan_videos()
        logger.info("Initial video scan completed")
    except Exception as e:
        logger.error(f"Initial video scan failed: {e}")
    
    # Cleanup old jobs
    try:
        removed = pipeline_executor.cleanup_old_jobs(days=7)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old pipeline jobs")
    except Exception as e:
        logger.error(f"Job cleanup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("VideoEmotion API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
