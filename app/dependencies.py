from functools import lru_cache
import sys

from src.core.video_manager import VideoManager
from src.core.scanner import VideoScanner
from src.core.metadata import MetadataStore
from src.core.stats import StatsCalculator
from src.core.trash_manager import TrashManager
from src.core.stats_updater import StatsUpdater
from src.core.pipeline_executor import PipelineExecutor
from src.core.realtime_manager import RealtimeManager
from app.config import settings


@lru_cache()
def get_video_manager() -> VideoManager:
    """
    Singleton factory for VideoManager.
    Initialized with the global VideoScanner, MetadataStore, and StatsCalculator.
    """
    # Manual Dependency Injection
    return VideoManager(
        project_root=settings.server.PROJECT_ROOT,
        scanner=VideoScanner(
            videos_dir=settings.server.PROJECT_ROOT / settings.project.paths.videos,
            realtime_dir=settings.server.PROJECT_ROOT
            / settings.project.paths.realtime_output,
        ),
        store=MetadataStore(
            metadata_path=settings.server.PROJECT_ROOT / settings.project.paths.metadata
        ),
        stats_calculator=StatsCalculator(),
    )


@lru_cache()
def get_trash_manager() -> TrashManager:
    """
    Singleton factory for TrashManager.
    Configured with project and trash root directories.
    """
    return TrashManager(
        project_root=settings.server.PROJECT_ROOT,
        trash_root=settings.server.PROJECT_ROOT / settings.project.paths.trash,
    )


@lru_cache()
def get_stats_updater() -> StatsUpdater:
    """
    Singleton factory for StatsUpdater.
    Used for background recalculation of statistics.
    """
    return StatsUpdater(project_root=settings.server.PROJECT_ROOT)


@lru_cache()
def get_pipeline_executor() -> PipelineExecutor:
    """
    Singleton factory for PipelineExecutor.
    Manages offline analysis jobs.
    """
    return PipelineExecutor(project_root=settings.server.PROJECT_ROOT)


@lru_cache()
def get_realtime_manager() -> RealtimeManager:
    """
    Singleton factory for RealtimeManager.
    Auto-detects 'mp_env' Python executable for running realtime scripts if available.
    """
    # Logic to select python from mp_env if exists
    mp_env_python = settings.server.PROJECT_ROOT / "mp_env" / "Scripts" / "python.exe"
    python_exec = str(mp_env_python) if mp_env_python.exists() else sys.executable

    return RealtimeManager(
        project_root=settings.server.PROJECT_ROOT, python_executable=python_exec
    )
