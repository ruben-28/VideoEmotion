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

    def __init__(
        self,
        project_root: Path,
        scanner: VideoScanner,
        store: MetadataStore,
        stats_calculator: StatsCalculator,
    ):
        """
        Initialize VideoManager with injected dependencies.

        Args:
            project_root: Root path of the project.
            scanner: Service to scan filesystem for videos.
            store: Service to persist metadata.
            stats_calculator: Service to calculate video statistics.
        """
        self.project_root = Path(project_root)
        self.scanner = scanner
        self.store = store
        self.stats_calculator = stats_calculator

        # We might still need an executor for internal background tasks if not strictly async everywhere
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _scan_logic(self) -> List[VideoMetadata]:
        """
        Internal synchronous logic to scan both offline and realtime videos.
        This serves as the single source of truth for scanning.
        """
        offline_paths = self.scanner.scan_offline()
        realtime_paths = self.scanner.scan_realtime()

        all_videos = []

        # Process Offline
        for p in offline_paths:
            try:
                all_videos.append(self._process_offline_video(p))
            except Exception as e:
                logger.error(f"Error processing offline video {p}: {e}")

        # Process Realtime
        for p in realtime_paths:
            try:
                all_videos.append(self._process_realtime_session(p))
            except Exception as e:
                logger.error(f"Error processing realtime session {p}: {e}")

        # Update Store & Prune Missing
        found_ids = {v.id for v in all_videos}
        existing_ids = list(self.store.list_videos().keys())

        for vid_id in existing_ids:
            if vid_id not in found_ids:
                self.store.delete_video(vid_id)

        for video in all_videos:
            self.store.set_video(video.id, video.to_dict())
        self.store.save()

        logger.info(f"Scanned {len(all_videos)} videos")
        return all_videos

    def scan_videos(self) -> List[VideoMetadata]:
        """Synchronous scan"""
        return self._scan_logic()

    async def scan_videos_async(self) -> List[VideoMetadata]:
        """
        Asynchronous scan wrapper.
        Runs the synchronous logic in a separate thread to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._scan_logic)

    def _process_offline_video(self, video_path: Path) -> VideoMetadata:
        """Process offline video path into metadata"""
        video_name = video_path.stem
        video_id = f"{video_name}_offline"

        reports_dir = self.project_root / "output" / "reports" / "offline" / video_name
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
            "reports": str(reports_dir),
            "extracted_frames": str(frames_dir),
            "detected_faces": str(self.project_root / "data" / "detected_faces" / video_name),
            "emotion_results": str(results_dir),
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
            stats=stats,
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

        file_paths = {"session_data": str(session_dir), "results": str(emotions_json)}

        # Calculate directory size
        total_size = sum(
            f.stat().st_size for f in session_dir.rglob("*") if f.is_file()
        )

        return VideoMetadata(
            id=video_id,
            name=session_name,
            mode=VideoMode.REALTIME,
            status=status,
            created_at=datetime.fromtimestamp(session_dir.stat().st_ctime),
            processed_at=processed_at,
            file_paths=file_paths,
            file_size_bytes=total_size,
            stats=stats,
        )

    def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        data = self.store.get_video(video_id)
        if data:
            return VideoMetadata.from_dict(data)
        return None

    def list_videos(
        self,
        mode: Optional[VideoMode] = None,
        status: Optional[VideoStatus] = None,
        sort_by="created_at",
        sort_order="desc",
    ) -> List[VideoMetadata]:
        raw_videos = self.store.list_videos().values()
        videos = [VideoMetadata.from_dict(v) for v in raw_videos]

        # Filter (Using robust comparison)
        if mode:
            target_mode = mode.value if hasattr(mode, "value") else str(mode)
            videos = [
                v
                for v in videos
                if (v.mode.value if hasattr(v.mode, "value") else str(v.mode))
                == target_mode
            ]
        if status:
            target_status = status.value if hasattr(status, "value") else str(status)
            videos = [
                v
                for v in videos
                if (v.status.value if hasattr(v.status, "value") else str(v.status))
                == target_status
            ]

        # Sort
        reverse = sort_order.lower() == "desc"
        if sort_by == "name":
            videos.sort(key=lambda v: v.name.lower(), reverse=reverse)
        elif sort_by == "status":
            videos.sort(key=lambda v: v.status.value, reverse=reverse)
        else:
            # Default to created_at
            videos.sort(key=lambda v: v.created_at, reverse=reverse)

        return videos

    def batch_get_videos_async(self, video_ids: List[str]):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            None, lambda: [self.get_video(v) for v in video_ids]
        )

    def update_video(self, video_id: str, updates: Dict) -> bool:
        """Update video metadata"""
        # Note: Direct metadata manipulation might bypass store in current architecture
        # Ideally, this should go through store methods
        # For refactor compatibility we keep broadly same logic but relying on store if possible
        # But 'updates' is a dict, so we rely on store's list_videos returning a ref or re-fetch

        # Current store implementation: list_videos returns a copy/dict
        # We need to explicitly save back
        if not self.store.update_video(video_id, updates):
            return False
        return True

    def delete_video_metadata(self, video_id: str) -> bool:
        return self.store.delete_video(video_id)

    def add_video_metadata(self, video: VideoMetadata) -> None:
        self.store.set_video(video.id, video.to_dict())
        self.store.save()

    def get_unprocessed_videos(self) -> List[VideoMetadata]:
        return self.list_videos(mode=VideoMode.OFFLINE, status=VideoStatus.UNPROCESSED)

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
            "emotion_distribution": self._calculate_global_emotions(videos),
        }

    def _calculate_global_emotions(
        self, videos: List[VideoMetadata]
    ) -> Dict[str, float]:
        from collections import defaultdict

        weighted_sums = defaultdict(float)
        total_weight = 0.0

        for v in videos:
            if not v.stats or not v.stats.get("global_distribution"):
                continue

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
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
