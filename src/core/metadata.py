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
        self.metadata_path = Path(metadata_path)
        self.data: Dict = {"version": "1.0", "videos": {}, "trash": {}}
        self.load()

    def load(self) -> None:
        """Load metadata from disk"""
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
        """Save metadata to disk"""
        self.data["last_updated"] = datetime.now().isoformat()
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def get_video(self, video_id: str) -> Optional[Dict]:
        return self.data["videos"].get(video_id)

    def set_video(self, video_id: str, data: Dict) -> None:
        self.data["videos"][video_id] = data

    def list_videos(self) -> Dict[str, Dict]:
        return self.data.get("videos", {})
