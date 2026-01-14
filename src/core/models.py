"""
Data models for VideoEmotion administration system.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Literal
from pathlib import Path


class VideoStatus(Enum):
    """Video processing status"""
    UNPROCESSED = "unprocessed"
    PARTIAL = "partial"
    PROCESSED = "processed"


class VideoMode(Enum):
    """Video processing mode"""
    OFFLINE = "offline"
    REALTIME = "realtime"


class JobStatus(Enum):
    """Pipeline job status"""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class VideoMetadata:
    """Metadata for a video entity"""
    id: str
    name: str
    mode: VideoMode
    status: VideoStatus
    created_at: datetime
    processed_at: Optional[datetime] = None
    file_paths: Dict[str, str] = field(default_factory=dict)
    pipeline_config: Optional[Dict] = None
    stats: Optional[Dict] = None
    file_size_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "file_paths": self.file_paths,
            "pipeline_config": self.pipeline_config,
            "stats": self.stats,
            "file_size_bytes": self.file_size_bytes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "VideoMetadata":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            mode=VideoMode(data["mode"]),
            status=VideoStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            processed_at=datetime.fromisoformat(data["processed_at"]) if data.get("processed_at") else None,
            file_paths=data.get("file_paths", {}),
            pipeline_config=data.get("pipeline_config"),
            stats=data.get("stats"),
            file_size_bytes=data.get("file_size_bytes"),
        )


@dataclass
class TrashMetadata:
    """Metadata for a trashed video"""
    trash_id: str
    original_video_id: str
    video_name: str
    mode: VideoMode
    original_status: VideoStatus
    deleted_at: datetime
    original_paths: Dict[str, str]
    trash_paths: Dict[str, str]
    size_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "trash_id": self.trash_id,
            "original_video_id": self.original_video_id,
            "video_name": self.video_name,
            "mode": self.mode.value,
            "original_status": self.original_status.value,
            "deleted_at": self.deleted_at.isoformat(),
            "original_paths": self.original_paths,
            "trash_paths": self.trash_paths,
            "size_bytes": self.size_bytes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TrashMetadata":
        """Create from dictionary"""
        return cls(
            trash_id=data["trash_id"],
            original_video_id=data["original_video_id"],
            video_name=data["video_name"],
            mode=VideoMode(data["mode"]),
            original_status=VideoStatus(data.get("original_status", "unprocessed")),
            deleted_at=datetime.fromisoformat(data["deleted_at"]),
            original_paths=data["original_paths"],
            trash_paths=data["trash_paths"],
            size_bytes=data.get("size_bytes"),
        )


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    fps: int = 5
    smoothing: bool = True
    tta: bool = True
    backend: Literal["hsemotion", "deepface"] = "hsemotion"
    visualize: bool = True
    overwrite: bool = False
    export_bboxes: bool = True
    no_extract: bool = False
    no_detect: bool = False
    no_analyze: bool = False
    no_summary: bool = False
    no_visualize: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineConfig":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class PipelineProgress:
    """Progress information for a pipeline job"""
    current_step: str
    current_step_index: int
    total_steps: int
    percent: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PipelineJob:
    """Pipeline execution job"""
    job_id: str
    video_name: str
    status: JobStatus
    config: PipelineConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[PipelineProgress] = None
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "job_id": self.job_id,
            "video_name": self.video_name,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress.to_dict() if self.progress else None,
            "logs": self.logs,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineJob":
        """Create from dictionary"""
        progress_data = data.get("progress")
        progress = PipelineProgress(**progress_data) if progress_data else None
        
        return cls(
            job_id=data["job_id"],
            video_name=data["video_name"],
            status=JobStatus(data["status"]),
            config=PipelineConfig.from_dict(data["config"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress=progress,
            logs=data.get("logs", []),
            error=data.get("error"),
        )
