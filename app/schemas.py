from typing import List, Optional
from pydantic import BaseModel


# Video Models
class VideoResponse(BaseModel):
    id: str
    name: str
    mode: str
    status: str
    created_at: str
    processed_at: Optional[str]
    file_size_mb: Optional[float]
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    stats: Optional[dict] = None


class VideoListResponse(BaseModel):
    videos: List[VideoResponse]
    total: int
    pagination: dict


# Trash Models
class TrashItemResponse(BaseModel):
    trash_id: str
    video_name: str
    mode: str
    original_status: str
    deleted_at: str
    size_mb: Optional[float]


# Pipeline Models
class PipelineRunRequest(BaseModel):
    video_name: str
    options: Optional[dict] = None


class PipelineJobResponse(BaseModel):
    job_id: str
    video_name: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: Optional[dict]
    logs: List[str]
    error: Optional[str]


# Stats Models
class StatsResponse(BaseModel):
    total_videos: int
    offline_videos: int
    realtime_videos: int
    processed: int
    partial: int
    unprocessed: int
    total_size_mb: float
    trash_stats: dict
    emotion_distribution: Optional[dict] = None


# Realtime Models
class RealtimeConfigRequest(BaseModel):
    camera_id: int = 0
    display_width: int = 800
    min_det_score: float = 0.65
    save_json: bool = True
    save_video: bool = True
    visualize: bool = True


class RealtimeStatusResponse(BaseModel):
    session_id: str
    status: str
    start_time: str
    config: dict
    output_dir: Optional[str] = None
    error: Optional[str] = None
