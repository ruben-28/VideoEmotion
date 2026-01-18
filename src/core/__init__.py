"""
VideoEmotion Core Module
Core functionality for video management, trash operations, and statistics.
"""

from .models import (
    VideoStatus,
    VideoMode,
    VideoMetadata,
    TrashMetadata,
    PipelineConfig,
    PipelineJob,
    JobStatus,
)
from .video_manager import VideoManager
from .trash_manager import TrashManager
from .stats_updater import StatsUpdater
from .pipeline_executor import PipelineExecutor
from .realtime_manager import RealtimeManager, RealtimeConfig, RealtimeSession, RealtimeStatus

__all__ = [
    "VideoStatus",
    "VideoMode",
    "VideoMetadata",
    "TrashMetadata",
    "PipelineConfig",
    "PipelineJob",
    "JobStatus",
    "VideoManager",
    "TrashManager",
    "StatsUpdater",
    "PipelineExecutor",
]

