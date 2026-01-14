
from pathlib import Path
from functools import lru_cache
from typing import Generator

from src.core.video_manager import VideoManager
from src.core.trash_manager import TrashManager
from src.core.stats_updater import StatsUpdater
from src.core.pipeline_executor import PipelineExecutor

# Hardcoded for now, ideal to move to config
PROJECT_ROOT = Path("c:/Users/ruben/Desktop/VideoEmotion")

@lru_cache()
def get_project_root() -> Path:
    return PROJECT_ROOT

@lru_cache()
def get_video_manager(
    root: Path = PROJECT_ROOT, # Hack to make it work without dependency chain for now
) -> VideoManager:
    return VideoManager(
        project_root=root,
        metadata_path=root / "video_metadata.json"
    )

@lru_cache()
def get_trash_manager(
    root: Path = PROJECT_ROOT
) -> TrashManager:
    return TrashManager(
        project_root=root,
        trash_root=root / "trash"
    )

@lru_cache()
def get_stats_updater(
    root: Path = PROJECT_ROOT
) -> StatsUpdater:
    return StatsUpdater(project_root=root)

@lru_cache()
def get_pipeline_executor(
    root: Path = PROJECT_ROOT
) -> PipelineExecutor:
    return PipelineExecutor(project_root=root)
