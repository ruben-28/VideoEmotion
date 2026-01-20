from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Any, Dict

from src.core.trash_manager import TrashManager
from src.core.video_manager import VideoManager
from src.core.stats_updater import StatsUpdater
from app.dependencies import get_trash_manager, get_video_manager, get_stats_updater
from app.schemas import TrashItemResponse

router = APIRouter(prefix="/api/trash", tags=["Trash"])


@router.get("", response_model=Dict[str, Any])
async def list_trash(trash_manager: TrashManager = Depends(get_trash_manager)):
    """
    List all pending items in the trash.

    Returns:
        dict: containing 'trash_items' list and 'total' count.
    """
    trash_items = trash_manager.list_trash()

    responses = []
    for item in trash_items:
        responses.append(
            TrashItemResponse(
                trash_id=item.trash_id,
                video_name=item.video_name,
                mode=item.mode.value,
                original_status=item.original_status.value,
                deleted_at=item.deleted_at.isoformat(),
                size_mb=round(item.size_bytes / (1024 * 1024), 2)
                if item.size_bytes
                else None,
            )
        )

    return {"trash_items": responses, "total": len(responses)}


@router.post("/{trash_id}/restore")
async def restore_video(
    trash_id: str,
    background_tasks: BackgroundTasks,
    trash_manager: TrashManager = Depends(get_trash_manager),
    video_manager: VideoManager = Depends(get_video_manager),
    stats_updater: StatsUpdater = Depends(get_stats_updater),
):
    """
    Restore a video from trash to its original location.

    Logic:
    1. Check availability of original path conflicts.
    2. Move files back.
    3. Trigger background re-scan and stats update.

    Args:
        trash_id (str): ID of the trash item.
        background_tasks (BackgroundTasks): Task manager.

    Returns:
        dict: Success message and restored video ID.

    Raises:
        HTTPException: If restoration fails.
    """
    success, original_video_id = trash_manager.restore_from_trash(trash_id)

    if success:
        # Re-scan to update metadata
        background_tasks.add_task(video_manager.scan_videos_async)
        background_tasks.add_task(stats_updater.recalculate_all_stats)

        return {
            "success": True,
            "message": "Video restored successfully",
            "video_id": original_video_id,
        }
    else:
        raise HTTPException(status_code=500, detail="Restoration failed")


@router.delete("/{trash_id}")
async def delete_permanently(
    trash_id: str, trash_manager: TrashManager = Depends(get_trash_manager)
):
    """
    Permanently delete a single item from the trash.
    THIS ACTION IS IRREVERSIBLE.

    Args:
        trash_id (str): ID of the trash item.

    Returns:
        dict: Success message and amount of space freed.
    """
    size_bytes = trash_manager.delete_permanently(trash_id)

    return {
        "success": True,
        "message": "Video permanently deleted",
        "freed_space_mb": round(size_bytes / (1024 * 1024), 2),
    }


@router.post("/empty")
async def empty_trash(trash_manager: TrashManager = Depends(get_trash_manager)):
    """
    Empty the entire trash (delete all items permanently).
    THIS ACTION IS IRREVERSIBLE.

    Returns:
        dict: Success message, count of deleted items, and total space freed.
    """
    count, total_size = trash_manager.empty_trash()

    return {
        "success": True,
        "message": f"Deleted {count} items",
        "freed_space_mb": round(total_size / (1024 * 1024), 2),
    }
