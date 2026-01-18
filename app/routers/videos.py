from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Optional

from src.core.video_manager import VideoManager
from src.core.trash_manager import TrashManager
from src.core.stats_updater import StatsUpdater
from src.core.exceptions import VideoNotFoundError
from src.core.models import VideoMode, VideoStatus
from app.dependencies import get_video_manager, get_trash_manager, get_stats_updater
from app.schemas import VideoListResponse, VideoResponse
from app.config import settings

router = APIRouter(prefix="/api/videos", tags=["Videos"])

@router.get("", response_model=VideoListResponse)
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
            # Determine URLs
            viz_h264 = settings.server.PROJECT_ROOT / settings.project.paths.visualizations / v.name / f"{v.name}_annotated_h264.mp4"
            viz_raw = settings.server.PROJECT_ROOT / settings.project.paths.visualizations / v.name / f"{v.name}_annotated_raw.mp4"
            realtime_h264 = settings.server.PROJECT_ROOT / settings.project.paths.realtime_output / v.name / "session_h264.mp4"
            
            # TODO: Move URL generation to a helper or schema method to handle static pathing cleaner
            if viz_h264.exists():
                video_url = f"/static/output/visualizations/{v.name}/{v.name}_annotated_h264.mp4"
            elif viz_raw.exists():
                 video_url = f"/static/output/visualizations/{v.name}/{v.name}_annotated_raw.mp4"
            elif realtime_h264.exists():
                 video_url = f"/static/output/realtime/{v.name}/session_h264.mp4"
            else:
                video_url = f"/static/videos/{v.name}/{v.name}.mp4"
            
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
        # In a real app we'd let global handler catch this, but for now specific log
        raise e

@router.get("/unprocessed")
async def list_unprocessed_videos(video_manager: VideoManager = Depends(get_video_manager)):
    """List unprocessed offline videos"""
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

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    video_manager: VideoManager = Depends(get_video_manager)
):
    """Get video details by ID"""
    video = video_manager.get_video(video_id)
    if not video:
        raise VideoNotFoundError(video_id)
    
    # URL Logic duplication... needs helper
    viz_h264 = settings.server.PROJECT_ROOT / settings.project.paths.visualizations / video.name / f"{video.name}_annotated_h264.mp4"
    viz_raw = settings.server.PROJECT_ROOT / settings.project.paths.visualizations / video.name / f"{video.name}_annotated_raw.mp4"
    realtime_h264 = settings.server.PROJECT_ROOT / settings.project.paths.realtime_output / video.name / "session_h264.mp4"
    
    if viz_h264.exists():
        video_url = f"/static/output/visualizations/{video.name}/{video.name}_annotated_h264.mp4"
    elif viz_raw.exists():
        video_url = f"/static/output/visualizations/{video.name}/{video.name}_annotated_raw.mp4"
    elif realtime_h264.exists():
        video_url = f"/static/output/realtime/{video.name}/session_h264.mp4"
    else:
        video_url = f"/static/videos/{video.name}/{video.name}.mp4"
    
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

@router.post("/scan")
async def scan_videos(
    background_tasks: BackgroundTasks,
    video_manager: VideoManager = Depends(get_video_manager)
):
    """Trigger video scan to update inventory"""
    background_tasks.add_task(video_manager.scan_videos_async)
    return {"message": "Video scan started", "status": "pending"}

@router.delete("/{video_id}")
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
        raise VideoNotFoundError(video_id)
    
    # Move to trash
    trash_meta = trash_manager.move_to_trash(video)
    
    # Update video metadata
    video_manager.scan_videos()
    
    # Recalculate stats in background
    background_tasks.add_task(stats_updater.recalculate_all_stats)
    
    return {
        "success": True,
        "message": "Video moved to trash",
        "trash_id": trash_meta.trash_id,
        "stats_updated": "pending"
    }
