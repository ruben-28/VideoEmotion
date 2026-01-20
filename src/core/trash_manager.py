"""
Trash Manager Module.

This module is responsible for safe deletion and restoration of video data.
It implements a "soft delete" mechanism where files are moved to a trash directory
instead of being immediately permanently deleted. It supports:
- Moving video files and related artifacts (frames, results) to trash.
- Restoring files from trash to their original locations.
- Permanently deleting files from trash.
- Batch operations with asynchronous support.
- Rollback mechanisms to ensure atomicity of move/restore operations.
"""

import shutil
import json
import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

from .models import VideoMetadata, TrashMetadata, VideoMode
from .emotion_cleaner import EmotionResultsCleaner

logger = logging.getLogger(__name__)


class TrashManager:
    """
    Manages trash operations for videos, ensuring safe deletion and restoration.

    Attributes:
        project_root (Path): Root directory of the project.
        trash_root (Path): Directory where trashed items are stored.
        cleaner (EmotionResultsCleaner): Utility to clean up partial analysis results.
    """

    def __init__(self, project_root: Path, trash_root: Path):
        """
        Initialize the TrashManager.

        Args:
            project_root (Path): Root path of the project.
            trash_root (Path): Path to the trash directory.
        """
        self.project_root = Path(project_root)
        self.trash_root = Path(trash_root)
        self.trash_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.cleaner = EmotionResultsCleaner(project_root)

    def _safe_rmtree(self, path: Path, retries: int = 5, delay: float = 0.5):
        """
        Robustly remove a directory tree with retries and permission handling.
        
        Logic:
        - Checks if path exists.
        - Attempts removal in a loop.
        - On failure, attempts to change file permissions (chmod +w).
        - Waits between retries.

        Args:
            path (Path): Directory or file to remove.
            retries (int): Number of retry attempts.
            delay (float): Seconds to wait between retries.
        """
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
                logger.warning(
                    f"Attempt {i + 1}/{retries} to remove {path} failed: {e}"
                )
                time.sleep(delay)

        # Final attempt
        if path.exists():
            shutil.rmtree(path, onerror=on_error)

    def move_to_trash(self, video_meta: VideoMetadata) -> TrashMetadata:
        """
        Move video and all related files to trash.

        Logic:
        1. Generates a unique trash ID based on video ID and timestamp.
        2. Creates a destination directory in the trash.
        3. Iterates through all file paths associated with the video.
        4. Moves each file/directory to the trash directory.
        5. Saves metadata about the operation to support restoration.
        6. Implements rollback if any move operation fails.

        Args:
            video_meta (VideoMetadata): Metadata of the video to delete.

        Returns:
            TrashMetadata: Metadata describing the trashed item.
            
        Raises:
            Exception: If moving files fails (after attempting rollback).
        """
        timestamp = int(datetime.now().timestamp())
        trash_id = f"{video_meta.id}_{timestamp}"

        # Create trash directory
        trash_dir = (
            self.trash_root / "offline" / trash_id
            if video_meta.mode == VideoMode.OFFLINE
            else self.trash_root / "realtime" / trash_id
        )

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
                    dir_size = sum(
                        f.stat().st_size for f in path.rglob("*") if f.is_file()
                    )
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
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(trash_metadata.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(
            f"Successfully moved to trash: {trash_id} ({total_size / (1024 * 1024):.2f} MB)"
        )
        return trash_metadata

    async def batch_move_to_trash_async(
        self, videos: List[VideoMetadata]
    ) -> List[TrashMetadata]:
        """
        Move multiple videos to trash asynchronously.

        Args:
            videos (List[VideoMetadata]): List of videos to delete.

        Returns:
            List[TrashMetadata]: List of successfully trashed items.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self._executor, self.move_to_trash, video)
            for video in videos
        ]
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
        """
        Restore video from trash to original location.

        Logic:
        1. Finds the trash directory by ID.
        2. Loads trash metadata to know original paths.
        3. Checks for conflicts (if original files already exist).
        4. Moves files back from trash to original locations.
        5. Deletes the trash directory upon success.
        6. Rolls back (deletes restored files) if restoration fails mid-way.

        Args:
            trash_id (str): ID of the trash item to restore.

        Returns:
            Tuple[bool, str]: Success status and original video ID.

        Raises:
            Exception: If trash not found, conflicts exist, or restoration fails.
        """
        # Find trash directory
        trash_dir = self._find_trash_dir(trash_id)
        if not trash_dir:
            raise Exception(f"Trash entry not found: {trash_id}")

        # Load trash metadata
        metadata_file = trash_dir / ".trash_metadata.json"
        if not metadata_file.exists():
            raise Exception(f"Trash metadata not found: {metadata_file}")

        with open(metadata_file, "r", encoding="utf-8") as f:
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
            raise Exception(
                f"Cannot restore: paths already exist: {', '.join(conflicts[:3])}"
            )

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
                            import os
                            import stat

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
            logger.warning(
                f"Restored successfully but failed to remove trash directory: {e}"
            )
            # Even if we fail to remove the directory, the files are restored.
            # Failure here means the trash list might still show it, but it's technically a success for the user data.

        return True, trash_meta.original_video_id

    # ... (keeping async batch restore) ...

    def delete_permanently(self, trash_id: str) -> int:
        """
        Permanently delete from trash.

        Logic:
        1. Finds the trash directory.
        2. Cleans up associated master entries (e.g. from global analysis results).
        3. Deletes the trash directory from disk.

        Args:
            trash_id (str): ID of the trash item.

        Returns:
            int: Number of bytes freed.
        """
        trash_dir = self._find_trash_dir(trash_id)
        if not trash_dir:
            raise Exception(f"Trash entry not found: {trash_id}")

        # Calculate size before deletion
        try:
            size_bytes = sum(
                f.stat().st_size for f in trash_dir.rglob("*") if f.is_file()
            )
        except:
            size_bytes = 0

        # Delete
        try:
            # Clean up master entries before deleting files
            self.cleaner.clean_master_entries(trash_dir)
            
            self._safe_rmtree(trash_dir)
            logger.info(
                f"Permanently deleted: {trash_id} ({size_bytes / (1024 * 1024):.2f} MB)"
            )
        except Exception as e:
            logger.error(f"Failed to permanently delete {trash_id}: {e}")
            raise

        return size_bytes

    async def batch_delete_permanently_async(self, trash_ids: List[str]) -> int:
        """
        Permanently delete multiple items from trash asynchronously.

        Args:
            trash_ids (List[str]): List of trash IDs.

        Returns:
            int: Total bytes freed.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self._executor, self.delete_permanently, tid)
            for tid in trash_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_freed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to delete {trash_ids[i]}: {result}")
            else:
                total_freed += result
        
        return total_freed

    def _clean_master_entries(self, trash_dir: Path) -> None:
        """
        Clean up entries from emotion_results_master.json derived from detected faces in this trash.
        This attempts to replicate the key generation logic from analyze_emotion.py.
        
        Args:
            trash_dir (Path): The directory of the trashed video.
        """
        # We need to find the deleted 'detected_faces' folder inside trash
        # Structure in trash: <trash_id>/detected_faces/...
        detected_faces_trash = trash_dir / "detected_faces"
        if not detected_faces_trash.exists():
            return

        master_json_path = self.project_root / "output" / "emotion_results" / "emotion_results_master.json"
        if not master_json_path.exists():
            return

        # Prepare list of keys to remove
        keys_to_remove = set()

        # Traverse the detected_faces in trash to reconstruct keys
        # Key format: relative path from faces_root
        # In trash, we have the content of the video folder.
        # But master keys are like "frames_fps5/person_0000/..."
        # We need to know what the relative path WAS.
        # However, Trash stores the COPY of the folder.
        # If we had: data/detected_faces/MyVideo/frames_fps5/...
        # In trash: trash_root/offline/<trash_id>/detected_faces/frames_fps5/...
        
        # So we can walk the trash detected_faces directory
        for root, _, files in os.walk(detected_faces_trash):
            for file in files:
                if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                
                # Full path in trash
                trash_file_path = Path(root) / file
                
                # Relative path from detected_faces folder in trash
                rel_path = trash_file_path.relative_to(detected_faces_trash)
                
                # The master json key is exactly this relative path? 
                # Wait. In analyze_emotion.py: 
                # rel_dir = os.path.relpath(dirpath, faces_root)
                # rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)
                #
                # faces_root usually points to data/detected_faces/<video>/frames_fpsX
                # OR data/detected_faces
                #
                # If pipeline calls analyze with faces_root = data/detected_faces/<video>/frames_fpsX
                # Then rel_dir starts from there.
                # BUT master keys usually look like: "frames_fps5\person_0000\frame_..."
                # Let's check the master json content again.
                # "frames_fps5\\person_0000\\frame_00000_t00000000track000.jpg"
                # This suggests faces_root was the VIDEO folder, not frames_fpsX?
                #
                # In pipeline.py:
                # detected_video_root = detected_root / video_name / frames_dir
                # analyze_emotions_incremental(faces_root=str(detected_video_root), ...)
                #
                # If faces_root passed to analyze is .../frames_fps5
                # Then os.walk(faces_root) -> rel_dir is relative to frames_fps5.
                # So if folder is person_0000 inside frames_fps5
                # rel_dir = person_0000
                # rel_path = person_0000/image.jpg.
                #
                # BUT the keys in the provided JSON show "frames_fps5\\person_0000\\..."
                # This contradicts pipeline.py call unless pipeline argument changed or I misread.
                #
                # Let's re-read pipeline.py call:
                # detected_video_root = detected_root / video_name / frames_dir
                # analyze_emotions_incremental(faces_root=str(detected_video_root), ...)
                #
                # If the key in JSON is "frames_fps5\person_0000\..." then either:
                # 1. faces_root was actually data/detected_faces/<video> (parent of frames_fps5)
                # 2. OR the key is constructed differently.
                
                # Looking at analyze_emotion.py again:
                # rel_dir = os.path.relpath(dirpath, faces_root)
                # rel_path = ... os.path.join(rel_dir, filename)
                
                # If the JSON has "frames_fps5...", then rel_dir MUST start with frames_fps5.
                # This implies faces_root must be the PARENT of frames_fps5.
                # 
                # In pipeline.py:
                # detected_video_root = detected_root / video_name / frames_dir 
                # This looks like it points DIRECTLY to frames_fps5.
                # 
                # Checking master json again...
                # "frames_fps5\\person_0000\\frame_..."
                #
                # Wait, if I am deleting the video, I want to remove ALL entries for that video.
                # If I walk the trash detected_faces, I see:
                # extracted means: data/detected_faces/<video_name> (from VideoManager update)
                # So detected_faces_trash contains the content of <video_name>.
                # So it has frames_fps5/person_0000/...
                #
                # So relative path from detected_faces_trash IS "frames_fps5/person_0000/..."
                # This matches the JSON keys!
                # 
                # So the logic is:
                key = str(rel_path).replace("\\", "/") # Normalize to forward slash just in case? 
                # The JSON has backslashes on Windows probably?
                # "frames_fps5\\person_0000\\..."
                # Let's try to match both separators or rely on what's in JSON. 
                # The safest is to try to match exact string or normalized.
                # Actually, JSON keys are strings. 
                # Let's rely on standard path str conversion but mindful of separators.
                
                keys_to_remove.add(str(rel_path))
                # Also add the one with backslashes if current os is not windows (unlikely here)
                # Or with forward slashes
                keys_to_remove.add(str(rel_path).replace("/", "\\"))
                keys_to_remove.add(str(rel_path).replace("\\", "/"))

        if not keys_to_remove:
            return

        # Load Lock Save 
        # (Naive implementation, race conditions possible but unlikely in single user desktop app)
        try:
            with open(master_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            original_len = len(data)
            for k in list(data.keys()):
                # We normalize key from json to check against our set
                # But our set has multiple variants.
                # Let's just check direct presence
                if k in keys_to_remove:
                    del data[k]
            
            if len(data) < original_len:
                with open(master_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                logger.info(f"Removed {original_len - len(data)} entries from master JSON for trash {trash_dir.name}")
        
        except Exception as e:
            logger.error(f"Failed to clean up master JSON: {e}")

    def list_trash(self) -> List[TrashMetadata]:
        """
        List all items in trash.

        Returns:
            List[TrashMetadata]: List of trash items sorted by deletion date (newest first).
        """
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
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        trash_meta_dict = json.load(f)
                    trash_meta = TrashMetadata.from_dict(trash_meta_dict)
                    trash_items.append(trash_meta)
                except Exception as e:
                    logger.error(
                        f"Failed to load trash metadata from {metadata_file}: {e}"
                    )

        # Sort by deletion date (newest first)
        trash_items.sort(key=lambda x: x.deleted_at, reverse=True)

        return trash_items

    def get_trash_item(self, trash_id: str) -> Optional[TrashMetadata]:
        """
        Get specific trash item by ID.

        Args:
            trash_id (str): ID of the trash item.

        Returns:
            Optional[TrashMetadata]: The trash item or None if not found.
        """
        trash_dir = self._find_trash_dir(trash_id)
        if not trash_dir:
            return None

        metadata_file = trash_dir / ".trash_metadata.json"
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                trash_meta_dict = json.load(f)
            return TrashMetadata.from_dict(trash_meta_dict)
        except Exception as e:
            logger.error(f"Failed to load trash metadata: {e}")
            return None

    def empty_trash(self) -> Tuple[int, int]:
        """
        Empty entire trash (delete all items permanently).

        Returns:
            Tuple[int, int]: Count of items deleted and total size freed in bytes.
        """
        trash_items = self.list_trash()
        count = len(trash_items)
        total_size = 0

        for item in trash_items:
            try:
                size = self.delete_permanently(item.trash_id)
                total_size += size
            except Exception as e:
                logger.error(f"Failed to delete {item.trash_id}: {e}")

        logger.info(
            f"Emptied trash: {count} items, {total_size / (1024 * 1024):.2f} MB freed"
        )
        return count, total_size

    def get_trash_stats(self) -> Dict:
        """
        Get statistics about the trash.

        Returns:
            Dict: Statistics including total items, count by mode, and total size.
        """
        trash_items = self.list_trash()

        total_items = len(trash_items)
        offline_count = sum(1 for item in trash_items if item.mode == VideoMode.OFFLINE)
        realtime_count = sum(
            1 for item in trash_items if item.mode == VideoMode.REALTIME
        )
        total_size = sum(item.size_bytes or 0 for item in trash_items)

        return {
            "total_items": total_items,
            "offline_items": offline_count,
            "realtime_items": realtime_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def _find_trash_dir(self, trash_id: str) -> Optional[Path]:
        """
        Find trash directory by ID.

        Algorithms:
        1. Fast path: check direct path <mode>/<trash_id>
        2. Fallback: search metadata in all folders if ID doesn't match folder name directly.

        Args:
            trash_id (str): ID to search for.

        Returns:
            Optional[Path]: Path to the trash directory or None.
        """
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
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("trash_id") == trash_id:
                            return trash_dir
                    except:
                        pass

                # Fallback: check if trash_id is in matching name format
                if trash_id in trash_dir.name:
                    return trash_dir

        return None

    def _rollback_move(
        self, moved_paths: Dict[str, str], original_paths: Dict[str, str]
    ) -> None:
        """
        Rollback partial move operation by moving files back to original locations.

        Args:
            moved_paths (Dict[str, str]): Map of key -> new path.
            original_paths (Dict[str, str]): Map of key -> original path.
        """
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
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
