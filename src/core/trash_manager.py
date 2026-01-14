"""
Trash Manager - Handles video deletion, restoration, and permanent deletion.
Supports batch operations and rollback on errors.
"""

import shutil
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

from .models import VideoMetadata, TrashMetadata, VideoMode

logger = logging.getLogger(__name__)


class TrashManager:
    """Manages trash operations for videos"""
    
    def __init__(self, project_root: Path, trash_root: Path):
        self.project_root = Path(project_root)
        self.trash_root = Path(trash_root)
        self.trash_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _safe_rmtree(self, path: Path, retries: int = 5, delay: float = 0.5):
        """Robustly remove a directory tree with retries"""
        import time
        import os
        import stat

        if not path.exists():
            return

        def on_error(func, p, exc_info):
            # Try to clear readonly bit
            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except Exception:
                pass

        for i in range(retries):
            try:
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path, onerror=on_error)
                
                # Verify it's gone
                if not path.exists():
                    return
            except Exception as e:
                logger.warning(f"Attempt {i+1}/{retries} to remove {path} failed: {e}")
                time.sleep(delay)
        
        # Final attempt
        if path.exists():
             shutil.rmtree(path, onerror=on_error)

    def move_to_trash(self, video_meta: VideoMetadata) -> TrashMetadata:
        """Move video and all related files to trash"""
        timestamp = int(datetime.now().timestamp())
        trash_id = f"{video_meta.id}_{timestamp}"
        
        # Create trash directory
        trash_dir = self.trash_root / "offline" / trash_id if video_meta.mode == VideoMode.OFFLINE else self.trash_root / "realtime" / trash_id
        
        trash_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Moving {video_meta.id} to trash: {trash_dir}")
        
        # Move all related files
        moved_paths = {}
        total_size = 0
        
        try:
            for key, path_str in video_meta.file_paths.items():
                if not path_str:
                    continue
                
                path = Path(path_str)
                if not path.exists():
                    logger.warning(f"Path does not exist, skipping: {path}")
                    continue
                
                dest = trash_dir / key
                
                if path.is_dir():
                    # Copy directory
                    shutil.copytree(path, dest, dirs_exist_ok=True)
                    # Calculate size
                    dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    total_size += dir_size
                    # Remove original
                    self._safe_rmtree(path)
                    logger.debug(f"Moved directory: {path} -> {dest}")
                else:
                    # Copy file
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    total_size += path.stat().st_size
                    # Remove original
                    path.unlink()
                    logger.debug(f"Moved file: {path} -> {dest}")
                
                moved_paths[key] = str(dest.resolve())
        
        except Exception as e:
            logger.error(f"Failed to move to trash: {e}")
            # Attempt rollback
            self._rollback_move(moved_paths, video_meta.file_paths)
            raise Exception(f"Failed to move to trash: {e}")
        
        # Create trash metadata
        trash_metadata = TrashMetadata(
            trash_id=trash_id,
            original_video_id=video_meta.id,
            video_name=video_meta.name,
            mode=video_meta.mode,
            original_status=video_meta.status,
            deleted_at=datetime.now(),
            original_paths=video_meta.file_paths,
            trash_paths=moved_paths,
            size_bytes=total_size,
        )
        
        # Write metadata file
        metadata_file = trash_dir / ".trash_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(trash_metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully moved to trash: {trash_id} ({total_size / (1024*1024):.2f} MB)")
        return trash_metadata
    
    async def batch_move_to_trash_async(self, videos: List[VideoMetadata]) -> List[TrashMetadata]:
        """Move multiple videos to trash asynchronously"""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(self._executor, self.move_to_trash, video) for video in videos]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        trash_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to trash {videos[i].id}: {result}")
            else:
                trash_items.append(result)
        
        return trash_items
    
    def restore_from_trash(self, trash_id: str) -> Tuple[bool, str]:
        """Restore video from trash to original location"""
        # Find trash directory
        trash_dir = self._find_trash_dir(trash_id)
        if not trash_dir:
            raise Exception(f"Trash entry not found: {trash_id}")
        
        # Load trash metadata
        metadata_file = trash_dir / ".trash_metadata.json"
        if not metadata_file.exists():
            raise Exception(f"Trash metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            trash_meta_dict = json.load(f)
        
        trash_meta = TrashMetadata.from_dict(trash_meta_dict)
        
        # Check if original paths are free
        conflicts = []
        for path_str in trash_meta.original_paths.values():
            if not path_str:
                continue
            path = Path(path_str)
            if path.exists():
                conflicts.append(str(path))
        
        if conflicts:
            raise Exception(f"Cannot restore: paths already exist: {', '.join(conflicts[:3])}")
        
        logger.info(f"Restoring from trash: {trash_id}")
        
        # Restore files
        restored_paths = {}
        try:
            for key, trash_path_str in trash_meta.trash_paths.items():
                trash_path = Path(trash_path_str)
                original_path = Path(trash_meta.original_paths[key])
                
                if not trash_path.exists():
                    logger.warning(f"Trash path does not exist, skipping: {trash_path}")
                    continue
                
                # Create parent directory
                original_path.parent.mkdir(parents=True, exist_ok=True)
                
                if trash_path.is_dir():
                    shutil.copytree(trash_path, original_path, dirs_exist_ok=True)
                    logger.debug(f"Restored directory: {trash_path} -> {original_path}")
                else:
                    shutil.copy2(trash_path, original_path)
                    logger.debug(f"Restored file: {trash_path} -> {original_path}")
                
                restored_paths[key] = str(original_path.resolve())
        
        except Exception as e:
            logger.error(f"Failed to restore: {e}")
            # Rollback - delete restored files
            for path_str in restored_paths.values():
                path = Path(path_str)
                try:
                    if path.exists():
                        if path.is_dir():
                            self._safe_rmtree(path)
                        else:
                            import os, stat
                            os.chmod(path, stat.S_IWRITE)
                            path.unlink()
                except Exception as rollback_err:
                    logger.error(f"Rollback failed for {path}: {rollback_err}")
            
            raise Exception(f"Failed to restore: {e}")
        
        # Delete trash directory
        try:
            self._safe_rmtree(trash_dir)
            logger.info(f"Successfully restored and removed trash: {trash_id}")
        except Exception as e:
            logger.warning(f"Restored successfully but failed to remove trash directory: {e}")
            # Even if we fail to remove the directory, the files are restored.
            # Failure here means the trash list might still show it, but it's technically a success for the user data.
        
        return True, trash_meta.original_video_id

    # ... (keeping async batch restore) ...

    def delete_permanently(self, trash_id: str) -> int:
        """Permanently delete from trash"""
        trash_dir = self._find_trash_dir(trash_id)
        if not trash_dir:
            raise Exception(f"Trash entry not found: {trash_id}")
        
        # Calculate size before deletion
        try:
            size_bytes = sum(f.stat().st_size for f in trash_dir.rglob('*') if f.is_file())
        except:
            size_bytes = 0
        
        # Delete
        try:
            self._safe_rmtree(trash_dir)
            logger.info(f"Permanently deleted: {trash_id} ({size_bytes / (1024*1024):.2f} MB)")
        except Exception as e:
            logger.error(f"Failed to permanently delete {trash_id}: {e}")
            raise
        
        return size_bytes
    
    async def batch_delete_permanently_async(self, trash_ids: List[str]) -> int:
        """Permanently delete multiple items from trash asynchronously"""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(self._executor, self.delete_permanently, tid) for tid in trash_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_freed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to delete {trash_ids[i]}: {result}")
            else:
                total_freed += result
        
        return total_freed
    
    def list_trash(self) -> List[TrashMetadata]:
        """List all items in trash"""
        trash_items = []
        
        if not self.trash_root.exists():
            return trash_items
        
        for mode_dir in self.trash_root.iterdir():
            if not mode_dir.is_dir():
                continue
            
            for trash_dir in mode_dir.iterdir():
                if not trash_dir.is_dir():
                    continue
                
                metadata_file = trash_dir / ".trash_metadata.json"
                if not metadata_file.exists():
                    logger.warning(f"Trash metadata missing: {trash_dir}")
                    continue
                
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        trash_meta_dict = json.load(f)
                    trash_meta = TrashMetadata.from_dict(trash_meta_dict)
                    trash_items.append(trash_meta)
                except Exception as e:
                    logger.error(f"Failed to load trash metadata from {metadata_file}: {e}")
        
        # Sort by deletion date (newest first)
        trash_items.sort(key=lambda x: x.deleted_at, reverse=True)
        
        return trash_items
    
    def get_trash_item(self, trash_id: str) -> Optional[TrashMetadata]:
        """Get trash item by ID"""
        trash_dir = self._find_trash_dir(trash_id)
        if not trash_dir:
            return None
        
        metadata_file = trash_dir / ".trash_metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                trash_meta_dict = json.load(f)
            return TrashMetadata.from_dict(trash_meta_dict)
        except Exception as e:
            logger.error(f"Failed to load trash metadata: {e}")
            return None
    
    def empty_trash(self) -> Tuple[int, int]:
        """Empty entire trash (delete all items permanently)"""
        trash_items = self.list_trash()
        count = len(trash_items)
        total_size = 0
        
        for item in trash_items:
            try:
                size = self.delete_permanently(item.trash_id)
                total_size += size
            except Exception as e:
                logger.error(f"Failed to delete {item.trash_id}: {e}")
        
        logger.info(f"Emptied trash: {count} items, {total_size / (1024*1024):.2f} MB freed")
        return count, total_size
    
    def get_trash_stats(self) -> Dict:
        """Get trash statistics"""
        trash_items = self.list_trash()
        
        total_items = len(trash_items)
        offline_count = sum(1 for item in trash_items if item.mode == VideoMode.OFFLINE)
        realtime_count = sum(1 for item in trash_items if item.mode == VideoMode.REALTIME)
        total_size = sum(item.size_bytes or 0 for item in trash_items)
        
        return {
            "total_items": total_items,
            "offline_items": offline_count,
            "realtime_items": realtime_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
    
    def _find_trash_dir(self, trash_id: str) -> Optional[Path]:
        """Find trash directory by ID"""
        if not self.trash_root.exists():
            return None
        
        # 1. Try finding directory directly (fast path for new items)
        for mode_dir in self.trash_root.iterdir():
            if not mode_dir.is_dir():
                continue
            
            direct_path = mode_dir / trash_id
            if direct_path.exists() and direct_path.is_dir():
                return direct_path
        
        # 2. Search by matching trash_id in metadata (fallback for legacy items)
        for mode_dir in self.trash_root.iterdir():
            if not mode_dir.is_dir():
                continue
            
            for trash_dir in mode_dir.iterdir():
                if not trash_dir.is_dir():
                    continue
                
                # Check metadata file
                metadata_file = trash_dir / ".trash_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if data.get("trash_id") == trash_id:
                            return trash_dir
                    except:
                        pass
                
                # Fallback: check if trash_id is in matching name format
                if trash_id in trash_dir.name:
                    return trash_dir

        return None
    
    def _rollback_move(self, moved_paths: Dict[str, str], original_paths: Dict[str, str]) -> None:
        """Rollback partial move operation"""
        logger.warning("Attempting rollback of partial move operation")
        
        for key, moved_path_str in moved_paths.items():
            moved_path = Path(moved_path_str)
            original_path = Path(original_paths[key])
            
            try:
                if moved_path.exists():
                    if moved_path.is_dir():
                        shutil.copytree(moved_path, original_path, dirs_exist_ok=True)
                    else:
                        original_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(moved_path, original_path)
                    logger.debug(f"Rolled back: {moved_path} -> {original_path}")
            except Exception as e:
                logger.error(f"Rollback failed for {key}: {e}")
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
