from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from src.core.video_manager import VideoManager
from src.core.trash_manager import TrashManager
from src.core.stats_updater import StatsUpdater
from app.dependencies import get_video_manager, get_trash_manager, get_stats_updater
from app.schemas import StatsResponse

router = APIRouter(prefix="/api/stats", tags=["Statistics"])

@router.get("", response_model=StatsResponse)
async def get_stats(
    video_manager: VideoManager = Depends(get_video_manager),
    trash_manager: TrashManager = Depends(get_trash_manager)
):
    """Get global statistics"""
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

@router.post("/refresh")
async def refresh_stats(
    background_tasks: BackgroundTasks,
    stats_updater: StatsUpdater = Depends(get_stats_updater)
):
    """Force statistics recalculation"""
    background_tasks.add_task(stats_updater.recalculate_all_stats)
    return {"message": "Statistics refresh started", "status": "pending"}
