from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Any, Dict

from src.core.trash_manager import TrashManager
from src.core.video_manager import VideoManager
from src.core.stats_updater import StatsUpdater
from app.dependencies import get_trash_manager, get_video_manager, get_stats_updater
from app.schemas import TrashItemResponse

router = APIRouter(prefix="/api/trash", tags=["Trash"])

@router.get("", response_model=Dict[str, Any])
async def list_trash(trash_manager: TrashManager = Depends(get_trash_manager)):
    """List all items in trash"""
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

@router.post("/{trash_id}/restore")
async def restore_video(
    trash_id: str, 
    background_tasks: BackgroundTasks,
    trash_manager: TrashManager = Depends(get_trash_manager),
    video_manager: VideoManager = Depends(get_video_manager),
    stats_updater: StatsUpdater = Depends(get_stats_updater)
):
    """Restore video from trash"""
    success, original_video_id = trash_manager.restore_from_trash(trash_id)
    
    if success:
        # Re-scan to update metadata
        background_tasks.add_task(video_manager.scan_videos_async)
        background_tasks.add_task(stats_updater.recalculate_all_stats)
        
        return {
            "success": True,
            "message": "Video restored successfully",
            "video_id": original_video_id
        }
    else:
        raise HTTPException(status_code=500, detail="Restoration failed")

@router.delete("/{trash_id}")
async def delete_permanently(
    trash_id: str,
    trash_manager: TrashManager = Depends(get_trash_manager)
):
    """Permanently delete from trash"""
    size_bytes = trash_manager.delete_permanently(trash_id)
    
    return {
        "success": True,
        "message": "Video permanently deleted",
        "freed_space_mb": round(size_bytes / (1024 * 1024), 2)
    }

@router.post("/empty")
async def empty_trash(
    trash_manager: TrashManager = Depends(get_trash_manager)
):
    """Empty entire trash"""
    count, total_size = trash_manager.empty_trash()
    
    return {
        "success": True,
        "message": f"Deleted {count} items",
        "freed_space_mb": round(total_size / (1024 * 1024), 2)
    }
