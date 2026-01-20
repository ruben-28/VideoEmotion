import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MetadataStore:
    """
    Responsible ONLY for loading and saving the global video metadata JSON.
    """

    def __init__(self, metadata_path: Path):
        """
        Initialize MetadataStore.

        Args:
            metadata_path (Path): Path to the JSON metadata file.
        """
        self.metadata_path = Path(metadata_path)
        self.data: Dict = {"version": "1.0", "videos": {}, "trash": {}}
        self.load()

    def load(self) -> None:
        """
        Load metadata from the JSON file on disk.
        If file doesn't exist, initializes a new one.
        Handles JSON errors gracefully by resetting to empty state.
        """
        if not self.metadata_path.exists():
            logger.info(f"Creating new metadata file: {self.metadata_path}")
            self.save()
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(f"Loaded metadata: {len(self.data.get('videos', {}))} videos")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.data = {"version": "1.0", "videos": {}, "trash": {}}

    def save(self) -> None:
        """
        Persist current metadata state to disk.
        Updates 'last_updated' timestamp.

        Raises:
            Exception: If writing to disk fails.
        """
        self.data["last_updated"] = datetime.now().isoformat()
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def get_video(self, video_id: str) -> Optional[Dict]:
        """Get raw dictionary for a video by ID."""
        return self.data["videos"].get(video_id)

    def set_video(self, video_id: str, data: Dict) -> None:
        """Set or update raw dictionary for a video."""
        self.data["videos"][video_id] = data

    def list_videos(self) -> Dict[str, Dict]:
        """Return the entire dictionary of video metadata."""
        return self.data.get("videos", {})

    def delete_video(self, video_id: str) -> bool:
        """
        Delete a video entry from metadata and save.

        Args:
            video_id (str): ID of the video to remove.

        Returns:
            bool: True if video was found and deleted, False otherwise.
        """
        if video_id in self.data["videos"]:
            del self.data["videos"][video_id]
            self.save()
            return True
        return False
