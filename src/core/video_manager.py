"""
Video Manager - Orchestrates video inventory using dedicated services.
"""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from .models import VideoMetadata, VideoMode, VideoStatus
from .scanner import VideoScanner
from .metadata import MetadataStore
from .stats import StatsCalculator

logger = logging.getLogger(__name__)

class VideoManager:
    """Orchestrator for video management"""
    
    def __init__(self, project_root: Path, metadata_path: Path):
        self.project_root = Path(project_root)
        
        # Initialize services
        # In a pure DI world, these would be injected into __init__, 
        # but for this refactor step we instantiate them here to keep call signature compatible 
        # (until we fix api.py).
        self.scanner = VideoScanner(project_root)
        self.store = MetadataStore(metadata_path)
        self.stats_calculator = StatsCalculator()
        
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def scan_videos_async(self) -> List[VideoMetadata]:
        """Scan filesystem asynchronously and update inventory"""
        loop = asyncio.get_event_loop()
        
        # Run scanning in thread pool
        offline_paths = await loop.run_in_executor(self._executor, self.scanner.scan_offline)
        realtime_paths = await loop.run_in_executor(self._executor, self.scanner.scan_realtime)
        
        # Process results
        all_videos = []
        
        # Process Offline
        for vid_path in offline_paths:
            try:
                meta = self._process_offline_video(vid_path)
                all_videos.append(meta)
            except Exception as e:
                logger.error(f"Error processing offline video {vid_path}: {e}")
                
        # Process Realtime
        for session_dir in realtime_paths:
            try:
                meta = self._process_realtime_session(session_dir)
                all_videos.append(meta)
            except Exception as e:
                logger.error(f"Error processing realtime session {session_dir}: {e}")
                
        # Update Store
        for video in all_videos:
            self.store.set_video(video.id, video.to_dict())
            
        self.store.save()
        logger.info(f"Scanned {len(all_videos)} videos")
        
        return all_videos
    
    def scan_videos(self) -> List[VideoMetadata]:
        """Synchronous scan"""
        offline_paths = self.scanner.scan_offline()
        realtime_paths = self.scanner.scan_realtime()
        
        all_videos = []
        for p in offline_paths:
             all_videos.append(self._process_offline_video(p))
        for p in realtime_paths:
             all_videos.append(self._process_realtime_session(p))
             
        for video in all_videos:
            self.store.set_video(video.id, video.to_dict())
        self.store.save()
        
        return all_videos
    
    def _process_offline_video(self, video_path: Path) -> VideoMetadata:
        """Process offline video path into metadata"""
        video_name = video_path.stem
        video_id = f"{video_name}_offline"
        
        # Reconstruct paths (Logic preserved but isolated here)
        # Ideally this path logic goes into a PathResolver service, but let's keep it here for now
        # to focus on the content refactor.
        reports_dir = self.project_root / "output" / "reports" / "offline" / video_name
        
        # Determine status (simplified check)
        # For full robustness, we should use the same logic as before, 
        # but for brevity in this refactor I'm trusting the existence of reports implies processed.
        # Actually, let's keep the exact logic if possible to be safe.
        frames_dir = self.project_root / "data" / "extracted_frames" / video_name
        results_dir = self.project_root / "output" / "emotion_results" / video_name
        
        frames_exist = frames_dir.exists() and any(frames_dir.iterdir())
        results_exist = results_dir.exists() and any(results_dir.iterdir())
        reports_exist = reports_dir.exists() and any(reports_dir.iterdir())
        
        if reports_exist:
            status = VideoStatus.PROCESSED
            processed_at = datetime.fromtimestamp(reports_dir.stat().st_mtime)
        elif results_exist or frames_exist:
            status = VideoStatus.PARTIAL
            processed_at = None
        else:
            status = VideoStatus.UNPROCESSED
            processed_at = None
            
        # Load Stats
        stats = self.stats_calculator.load_offline_stats(reports_dir)
        
        file_paths = {
            "video": str(video_path),
            "reports": str(reports_dir)
        }
        
        return VideoMetadata(
            id=video_id,
            name=video_name,
            mode=VideoMode.OFFLINE,
            status=status,
            created_at=datetime.fromtimestamp(video_path.stat().st_ctime),
            processed_at=processed_at,
            file_paths=file_paths,
            file_size_bytes=video_path.stat().st_size,
            stats=stats
        )

    def _process_realtime_session(self, session_dir: Path) -> VideoMetadata:
        session_name = session_dir.name
        video_id = f"{session_name}_realtime"
        emotions_json = session_dir / "realtime_emotions.json"
        
        if emotions_json.exists():
            status = VideoStatus.PROCESSED
            processed_at = datetime.fromtimestamp(emotions_json.stat().st_mtime)
            stats = self.stats_calculator.calculate_realtime_stats(emotions_json)
        else:
            status = VideoStatus.PARTIAL
            processed_at = None
            stats = None
            
        file_paths = {
            "session_data": str(session_dir),
            "results": str(emotions_json)
        }
        
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
        
        return VideoMetadata(
            id=video_id,
            name=session_name,
            mode=VideoMode.REALTIME,
            status=status,
            created_at=datetime.fromtimestamp(session_dir.stat().st_ctime),
            processed_at=processed_at,
            file_paths=file_paths,
            file_size_bytes=total_size,
            stats=stats
        )

    def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        data = self.store.get_video(video_id)
        if data:
            return VideoMetadata.from_dict(data)
        return None
    
    def list_videos(self, mode=None, status=None, sort_by="created_at", sort_order="desc") -> List[VideoMetadata]:
        raw_videos = self.store.list_videos().values()
        videos = [VideoMetadata.from_dict(v) for v in raw_videos]
        
        # Filter
        if mode:
            target_mode = mode.value if hasattr(mode, "value") else str(mode)
            videos = [
                v for v in videos 
                if (v.mode.value if hasattr(v.mode, "value") else str(v.mode)) == target_mode
            ]
        if status:
            target_status = status.value if hasattr(status, "value") else str(status)
            videos = [
                v for v in videos 
                if (v.status.value if hasattr(v.status, "value") else str(v.status)) == target_status
            ]
            
        # Sort
        reverse = (sort_order == "desc")
        if sort_by == "name":
            videos.sort(key=lambda v: v.name.lower(), reverse=reverse)
        elif sort_by == "status":
            videos.sort(key=lambda v: v.status.value, reverse=reverse)
        else:
            videos.sort(key=lambda v: v.created_at, reverse=reverse)
            
        return videos

    def batch_get_videos_async(self, video_ids: List[str]):
         # simple wrapper
         loop = asyncio.get_event_loop()
         return loop.run_in_executor(None, lambda: [self.get_video(v) for v in video_ids])
    
    def update_video(self, video_id: str, updates: Dict) -> bool:
        """Update video metadata"""
        if video_id not in self.metadata["videos"]:
            return False
        
        video_data = self.metadata["videos"][video_id]
        video_data.update(updates)
        self._save_metadata()
        return True
    
    def delete_video_metadata(self, video_id: str) -> bool:
        """Remove video from metadata (used when moving to trash)"""
        if video_id not in self.metadata["videos"]:
            return False
        
        del self.metadata["videos"][video_id]
        self._save_metadata()
        logger.info(f"Deleted video metadata: {video_id}")
        return True
    
    def add_video_metadata(self, video: VideoMetadata) -> None:
        """Add or update video metadata"""
        self.metadata["videos"][video.id] = video.to_dict()
        self._save_metadata()
        logger.info(f"Added video metadata: {video.id}")
    
    def get_unprocessed_videos(self) -> List[VideoMetadata]:
        """Get list of unprocessed offline videos"""
        return self.list_videos(
            mode=VideoMode.OFFLINE,
            status=VideoStatus.UNPROCESSED
        )
    
    def get_stats(self) -> Dict:
        """Get global statistics"""
        videos = self.list_videos()
        
        total = len(videos)
        offline_count = sum(1 for v in videos if v.mode == VideoMode.OFFLINE)
        realtime_count = sum(1 for v in videos if v.mode == VideoMode.REALTIME)
        
        processed = sum(1 for v in videos if v.status == VideoStatus.PROCESSED)
        partial = sum(1 for v in videos if v.status == VideoStatus.PARTIAL)
        unprocessed = sum(1 for v in videos if v.status == VideoStatus.UNPROCESSED)
        
        total_size = sum(v.file_size_bytes or 0 for v in videos)
        
        return {
            "total_videos": total,
            "offline_videos": offline_count,
            "realtime_videos": realtime_count,
            "processed": processed,
            "partial": partial,
            "unprocessed": unprocessed,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "emotion_distribution": self._calculate_global_emotions(videos)
        }

    def _calculate_global_emotions(self, videos: List[VideoMetadata]) -> Dict[str, float]:
        """Calculate global emotion distribution weighted by video duration"""
        from collections import defaultdict
        
        weighted_sums = defaultdict(float)
        total_weight = 0.0
        
        for v in videos:
            if not v.stats or not v.stats.get("global_distribution"):
                continue
            
            # Use total_frames as weight, default to 1 if missing but stats exist
            weight = v.stats.get("total_frames", 0)
            if weight == 0: 
                continue
                
            dist = v.stats["global_distribution"]
            for emo, score in dist.items():
                weighted_sums[emo] += score * weight
            
            total_weight += weight
            
        if total_weight == 0:
            return {}
            
        return {k: round(v / total_weight, 4) for k, v in weighted_sums.items()}
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
