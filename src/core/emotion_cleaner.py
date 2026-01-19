"""
Emotion Results Cleaner - Handles cleanup of master emotion results.
"""

import json
import os
import logging
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)


class EmotionResultsCleaner:
    """Service to clean up emotion results from the master JSON file."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.master_json_path = (
            self.project_root / "output" / "emotion_results" / "emotion_results_master.json"
        )

    def clean_master_entries(self, trash_dir: Path) -> None:
        """
        Clean up entries from emotion_results_master.json derived from detected faces in this trash.
        """
        # We need to find the deleted 'detected_faces' folder inside trash
        # Structure in trash: <trash_id>/detected_faces/...
        detected_faces_trash = trash_dir / "detected_faces"
        if not detected_faces_trash.exists():
            return

        if not self.master_json_path.exists():
            return

        # Prepare list of keys to remove
        keys_to_remove = set()

        # Traverse the detected_faces in trash to reconstruct keys
        for root, _, files in os.walk(detected_faces_trash):
            for file in files:
                if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                # Full path in trash
                trash_file_path = Path(root) / file

                # Relative path from detected_faces folder in trash
                try:
                    rel_path = trash_file_path.relative_to(detected_faces_trash)
                except ValueError:
                    continue

                # Add potential key variants
                keys_to_remove.add(str(rel_path))
                keys_to_remove.add(str(rel_path).replace("/", "\\"))
                keys_to_remove.add(str(rel_path).replace("\\", "/"))

        if not keys_to_remove:
            return

        try:
            with open(self.master_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            original_len = len(data)
            # Remove keys
            for k in list(data.keys()):
                if k in keys_to_remove:
                    del data[k]

            if len(data) < original_len:
                with open(self.master_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                logger.info(
                    f"Removed {original_len - len(data)} entries from master JSON for trash {trash_dir.name}"
                )

        except Exception as e:
            logger.error(f"Failed to clean up master JSON: {e}")
