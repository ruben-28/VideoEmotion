
from pathlib import Path
from typing import List, Set, Optional
import logging
from .models import VideoMetadata, VideoMode, VideoStatus

logger = logging.getLogger(__name__)

class VideoScanner:
    """
    Responsible ONLY for scanning the filesystem and identifying video files.
    """
    def __init__(self, videos_dir: Path, realtime_dir: Path):
        self.videos_dir = Path(videos_dir)
        self.realtime_dir = Path(realtime_dir)
        self.video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    def scan_offline(self) -> List[Path]:
        """Scan data/videos for video files"""
        videos_dir = self.videos_dir
        found_videos = []
        
        if not videos_dir.exists():
            return []

        # 1. Scan subdirectories
        for item in videos_dir.iterdir():
            if item.is_dir():
                # Prefer video with same name as folder
                possible_video = item / f"{item.name}.mp4"
                if possible_video.exists():
                    found_videos.append(possible_video)
                    continue
                
                # Else check extensions
                for ext in self.video_extensions:
                    candidate = item / f"{item.name}{ext}"
                    if candidate.exists():
                        found_videos.append(candidate)
                        break

        # 2. Scan root files (legacy)
        for video_file in videos_dir.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in self.video_extensions:
                if "_h264" not in video_file.stem:
                    found_videos.append(video_file)
                    
        return found_videos

    def scan_realtime(self) -> List[Path]:
        """Scan output/realtime for session directories"""
        realtime_dir = self.realtime_dir
        found_sessions = []
        
        if not realtime_dir.exists():
            return []
            
        for session_dir in realtime_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                found_sessions.append(session_dir)
                
        return found_sessions
