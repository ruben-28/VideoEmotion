"""
Video Manager - Handles video inventory, scanning, and metadata tracking.
Supports both offline and realtime video modes with async operations.
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import logging

from .models import VideoMetadata, VideoMode, VideoStatus

logger = logging.getLogger(__name__)


class VideoManager:
    """Manages video inventory and metadata"""
    
    def __init__(self, project_root: Path, metadata_path: Path):
        self.project_root = Path(project_root)
        self.metadata_path = Path(metadata_path)
        self.metadata: Dict = {"version": "1.0", "videos": {}, "trash": {}}
        self._load_metadata()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _load_metadata(self) -> None:
        """Load metadata from JSON file"""
        if not self.metadata_path.exists():
            logger.info(f"Creating new metadata file: {self.metadata_path}")
            self._save_metadata()
            return
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata: {len(self.metadata.get('videos', {}))} videos")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = {"version": "1.0", "videos": {}, "trash": {}}
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.debug("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    async def scan_videos_async(self) -> List[VideoMetadata]:
        """Scan filesystem asynchronously and build video inventory"""
        loop = asyncio.get_event_loop()
        
        # Scan offline and realtime in parallel
        offline_task = loop.run_in_executor(self._executor, self._scan_offline_videos)
        realtime_task = loop.run_in_executor(self._executor, self._scan_realtime_sessions)
        
        offline_videos, realtime_videos = await asyncio.gather(offline_task, realtime_task)
        
        all_videos = offline_videos + realtime_videos
        
        # Update metadata
        for video in all_videos:
            self.metadata["videos"][video.id] = video.to_dict()
        
        self._save_metadata()
        logger.info(f"Scanned {len(all_videos)} videos ({len(offline_videos)} offline, {len(realtime_videos)} realtime)")
        
        return all_videos
    
    def scan_videos(self) -> List[VideoMetadata]:
        """Synchronous version of scan_videos_async"""
        offline_videos = self._scan_offline_videos()
        realtime_videos = self._scan_realtime_sessions()
        
        all_videos = offline_videos + realtime_videos
        
        for video in all_videos:
            self.metadata["videos"][video.id] = video.to_dict()
        
        self._save_metadata()
        logger.info(f"Scanned {len(all_videos)} videos")
        
        return all_videos
    
    def _scan_offline_videos(self) -> List[VideoMetadata]:
        """Scan offline videos in data/videos"""
        videos = []
        videos_dir = self.project_root / "data" / "videos"
        
        if not videos_dir.exists():
            logger.warning(f"Videos directory not found: {videos_dir}")
            return videos
        
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        
        # Scan subdirectories (new structure)
        for item in videos_dir.iterdir():
            if item.is_dir():
                # Look for video file with same name inside
                # e.g. videos/myvideo/myvideo.mp4
                possible_video = item / f"{item.name}.mp4" # Try mp4 first
                
                # Search for any valid extension if specific logic is needed, 
                # but enforcing name match is good practice.
                found_video = None
                for ext in video_extensions:
                    candidate = item / f"{item.name}{ext}"
                    if candidate.exists():
                        found_video = candidate
                        break
                
                if found_video:
                    try:
                        video_meta = self._scan_offline_video(found_video)
                        videos.append(video_meta)
                    except Exception as e:
                        logger.error(f"Failed to scan {found_video.name}: {e}")

        # Scan root files (legacy structure)
        for video_file in videos_dir.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                # Ignore generated files in root
                if "_h264" in video_file.stem:
                    continue
                    
                try:
                    video_meta = self._scan_offline_video(video_file)
                    videos.append(video_meta)
                except Exception as e:
                    logger.error(f"Failed to scan {video_file.name}: {e}")
        
        return videos
    
    def _scan_offline_video(self, video_path: Path) -> VideoMetadata:
        """Scan a single offline video and determine processing status"""
        video_name = video_path.stem
        video_id = f"{video_name}_offline"
        
        # Define expected paths
        frames_dir = self.project_root / "data" / "extracted_frames" / video_name
        faces_dir = self.project_root / "data" / "detected_faces" / video_name
        results_dir = self.project_root / "output" / "emotion_results" / video_name
        reports_dir = self.project_root / "output" / "reports" / "offline" / video_name
        viz_dir = self.project_root / "output" / "visualizations" / video_name
        
        # Check existence
        frames_exist = frames_dir.exists() and any(frames_dir.iterdir())
        faces_exist = faces_dir.exists() and any(faces_dir.iterdir())
        results_exist = results_dir.exists() and any(results_dir.iterdir())
        reports_exist = reports_dir.exists() and any(reports_dir.iterdir())
        viz_exist = viz_dir.exists() and any(viz_dir.iterdir())
        
        # Determine status
        if all([frames_exist, faces_exist, results_exist, reports_exist, viz_exist]):
            status = VideoStatus.PROCESSED
            processed_at = datetime.fromtimestamp(reports_dir.stat().st_mtime)
        elif any([frames_exist, faces_exist, results_exist]):
            status = VideoStatus.PARTIAL
            processed_at = None
        else:
            status = VideoStatus.UNPROCESSED
            processed_at = None
        
        # Build file paths
        file_paths = {
            "video": str(video_path.resolve()),
            "frames": str(frames_dir.resolve()),
            "faces": str(faces_dir.resolve()),
            "results": str(results_dir.resolve()),
            "reports": str(reports_dir.resolve()),
            "visualizations": str(viz_dir.resolve()),
        }
        
        # Get file size
        try:
            file_size_bytes = video_path.stat().st_size
        except:
            file_size_bytes = None
            
        # Load stats from summary.json if available
        stats = None
        if reports_exist:
            try:
                # Scan recursively for summary.json (handles extra levels like frames_fps5)
                summaries = list(reports_dir.rglob("summary.json"))
                if summaries:
                    latest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
                    with open(latest_summary, "r", encoding="utf-8") as f:
                        stats = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load stats for {video_name}: {e}")
        
        return VideoMetadata(
            id=video_id,
            name=video_name,
            mode=VideoMode.OFFLINE,
            status=status,
            created_at=datetime.fromtimestamp(video_path.stat().st_ctime),
            processed_at=processed_at,
            file_paths=file_paths,
            file_size_bytes=file_size_bytes,
            stats=stats,
        )
    
    def _scan_realtime_sessions(self) -> List[VideoMetadata]:
        """Scan realtime sessions in output/realtime"""
        videos = []
        realtime_dir = self.project_root / "output" / "realtime"
        
        if not realtime_dir.exists():
            logger.warning(f"Realtime directory not found: {realtime_dir}")
            return videos
        
        for session_dir in realtime_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                try:
                    video_meta = self._scan_realtime_session(session_dir)
                    videos.append(video_meta)
                except Exception as e:
                    logger.error(f"Failed to scan {session_dir.name}: {e}")
        
        return videos
    
    def _scan_realtime_session(self, session_dir: Path) -> VideoMetadata:
        """Scan a single realtime session"""
        session_name = session_dir.name
        video_id = f"{session_name}_realtime"
        
        emotions_json = session_dir / "realtime_emotions.json"
        video_h264 = session_dir / "session_h264.mp4"
        video_raw = session_dir / "session.mp4"
        
        # Determine status
        if emotions_json.exists():
            status = VideoStatus.PROCESSED
            processed_at = datetime.fromtimestamp(emotions_json.stat().st_mtime)
        else:
            status = VideoStatus.PARTIAL
            processed_at = None
        
        # Choose video file
        video_file = video_h264 if video_h264.exists() else (video_raw if video_raw.exists() else None)
        
        file_paths = {
            "session_data": str(session_dir.resolve()),
            "video": str(video_file.resolve()) if video_file else "",
            "results": str(emotions_json.resolve()) if emotions_json.exists() else "",
        }
        
        # Calculate total size
        try:
            file_size_bytes = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
        except:
            file_size_bytes = None
        
        
        # Calculate stats from realtime_emotions.json
        stats = None
        if emotions_json.exists():
            try:
                stats = self._calculate_realtime_stats(emotions_json)
            except Exception as e:
                logger.warning(f"Failed to calculate stats for realtime session {session_name}: {e}")

        return VideoMetadata(
            id=video_id,
            name=session_name,
            mode=VideoMode.REALTIME,
            status=status,
            created_at=datetime.fromtimestamp(session_dir.stat().st_ctime),
            processed_at=processed_at,
            file_paths=file_paths,
            file_size_bytes=file_size_bytes,
            stats=stats,
        )

    def _calculate_realtime_stats(self, json_path: Path) -> Dict:
        """Calculate stats from a realtime_emotions.json file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        records = data.get("records", [])
        if not records:
            return None
            
        # 1. Normalize and aggregate
        from collections import Counter, defaultdict
        
        emotion_counts = Counter()
        timeline_buckets = defaultdict(Counter)
        
        valid_records = 0
        
        for rec in records:
            emo = rec.get("emotion")
            if not emo:
                continue
                
            # Normalize to lowercase
            emo = emo.strip().lower()
            
            emotion_counts[emo] += 1
            valid_records += 1
            
            # Bucket by second
            # time_ms is typically absolute or relative? 
            # The 't_rel_ms' seems to be relative to start based on cat output (50000ms = 50s)
            # Use t_rel_ms if available, else time_ms (which seemed large in cat output, likely epoch)
            # Actually, looking at cat output: time_ms: 2439998 (~40 mins?), t_rel_ms: 50000 (50s)
            # We should probably use t_rel_ms for the timeline relative to video start.
            
            t_ms = rec.get("t_rel_ms", rec.get("time_ms", 0))
            if t_ms is None: t_ms = 0
                
            sec = int(t_ms / 1000)
            timeline_buckets[sec][emo] += 1
            
        if valid_records == 0:
            return None
            
        # 2. Global Distribution
        global_distribution = {k: v / valid_records for k, v in emotion_counts.items()}
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # 3. Timeline
        timeline = []
        if timeline_buckets:
            max_sec = max(timeline_buckets.keys())
            for s in range(max_sec + 1):
                counts = timeline_buckets.get(s, Counter())
                total = sum(counts.values())
                if total > 0:
                    dist = {k: round(v / total, 3) for k, v in counts.items()}
                else:
                    dist = {}
                
                timeline.append({
                    "timestamp": s,
                    "emotions": dist
                })
        
        return {
            "global_distribution": global_distribution,
            "dominant_emotion": dominant_emotion,
            "timeline": timeline,
            # Add avg_emotion if needed, or simple placeholders
            "avg_emotion": global_distribution 
        }
    
    def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        """Get video by ID"""
        video_data = self.metadata["videos"].get(video_id)
        if not video_data:
            return None
        return VideoMetadata.from_dict(video_data)
    
    def list_videos(
        self,
        mode: Optional[VideoMode] = None,
        status: Optional[VideoStatus] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[VideoMetadata]:
        """List videos with optional filters and sorting"""
        videos = []
        
        for video_data in self.metadata["videos"].values():
            video = VideoMetadata.from_dict(video_data)
            
            # Apply filters
            if mode and video.mode != mode:
                continue
            if status and video.status != status:
                continue
            
            videos.append(video)
        
        # Sort
        reverse = (sort_order == "desc")
        if sort_by == "name":
            videos.sort(key=lambda v: v.name.lower(), reverse=reverse)
        elif sort_by == "status":
            videos.sort(key=lambda v: v.status.value, reverse=reverse)
        else:  # created_at (default)
            videos.sort(key=lambda v: v.created_at, reverse=reverse)
        
        return videos
    
    async def batch_get_videos_async(self, video_ids: List[str]) -> List[Optional[VideoMetadata]]:
        """Get multiple videos asynchronously"""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(self._executor, self.get_video, vid) for vid in video_ids]
        return await asyncio.gather(*tasks)
    
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
