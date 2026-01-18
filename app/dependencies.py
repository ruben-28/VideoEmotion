
from pathlib import Path
from functools import lru_cache
from typing import Generator

from src.core.video_manager import VideoManager
from src.core.trash_manager import TrashManager
from src.core.stats_updater import StatsUpdater
from src.core.pipeline_executor import PipelineExecutor
from src.core.realtime_manager import RealtimeManager

# Hardcoded for now, ideal to move to config
PROJECT_ROOT = Path("c:/Users/ruben/Desktop/VideoEmotion")

@lru_cache()
def get_project_root() -> Path:
    return PROJECT_ROOT

from app.config import settings

@lru_cache()
def get_video_manager(
    root: Path = PROJECT_ROOT, # Hack to make it work without dependency chain for now
) -> VideoManager:
    return VideoManager(
        project_root=root,
        metadata_path=root / "video_metadata.json",
        videos_dir=root / settings.paths.videos,
        realtime_dir=root / settings.paths.realtime_output
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

@lru_cache()
def get_realtime_manager(
    root: Path = PROJECT_ROOT
) -> RealtimeManager:
    # Check for mp_env specifically for realtime analysis
    mp_env_python = root / "mp_env" / "Scripts" / "python.exe"
    if mp_env_python.exists():
        python_exe = str(mp_env_python)
    else:
        import sys
        python_exe = sys.executable

    return RealtimeManager(project_root=root, python_executable=python_exe)
