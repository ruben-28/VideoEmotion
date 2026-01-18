"""
Path Resolver - Centralized path resolution for video files and metadata.
Handles finding latest sessions, video files, and validating file existence.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PathResolver:
    """Centralized path resolution for video files"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.data_videos = self.project_root / "data" / "videos"
        self.output_viz = self.project_root / "output" / "visualizations"
        self.output_reports_offline = (
            self.project_root / "output" / "reports" / "offline"
        )
        self.output_realtime = self.project_root / "output" / "realtime"

    def find_latest_session(
        self, video_name: str, mode: str = "offline"
    ) -> Optional[Path]:
        """
        Find the latest session directory for a video.

        Args:
            video_name: Name of the video (without extension)
            mode: "offline" or "realtime"

        Returns:
            Path to latest session directory, or None if not found
        """
        if mode == "offline":
            # Look in output/reports/offline/{video_name}
            base_dir = self.output_reports_offline / video_name

            if not base_dir.exists():
                return None

            # Find all potential session directories (any dir containing summary.json)
            # We search recursively to handle cases with or without intermediate folders like frames_fps5
            candidates = []
            for path in base_dir.rglob("summary.json"):
                session_dir = path.parent
                candidates.append(session_dir)

            if not candidates:
                return None

            # Sort by directory name (timestamp) descending
            latest = sorted(candidates, key=lambda p: p.name, reverse=True)[0]
            logger.info(f"Found latest session for {video_name}: {latest.name}")
            return latest

        elif mode == "realtime":
            # Realtime sessions are in output/realtime/{session_name}/
            session_dir = self.output_realtime / video_name
            return session_dir if session_dir.exists() else None

        return None

    def get_video_paths(
        self, video_name: str, mode: str = "offline"
    ) -> Dict[str, Optional[Path]]:
        """
        Get all relevant paths for a video.

        Args:
            video_name: Name of the video (without extension)
            mode: "offline" or "realtime"

        Returns:
            Dictionary with paths: video, summary, session_dir
        """
        paths = {
            "video": None,
            "summary": None,
            "session_dir": None,
        }

        if mode == "offline":
            # Find video file in visualizations
            viz_dir = self.output_viz / video_name
            video_candidates = [
                viz_dir / f"{video_name}_annotated_h264.mp4",
                viz_dir / f"{video_name}_annotated_raw.mp4",
                viz_dir / f"{video_name}_annotated_bbox_h264.mp4",
                viz_dir / f"{video_name}_annotated_bbox.mp4",
                # New structure
                self.data_videos / video_name / f"{video_name}_h264_unannotated.mp4",
                self.data_videos / video_name / f"{video_name}.mp4",
                self.data_videos / video_name / f"{video_name}.avi",
                self.data_videos / video_name / f"{video_name}.mov",
                self.data_videos / video_name / f"{video_name}.mkv",
                # Fallback to legacy structure
                self.data_videos / f"{video_name}.mp4",
                self.data_videos / f"{video_name}.avi",
                self.data_videos / f"{video_name}.mov",
            ]

            for candidate in video_candidates:
                if candidate.exists():
                    paths["video"] = candidate
                    break

            # Find latest session and summary
            latest_session = self.find_latest_session(video_name, mode)
            if latest_session:
                paths["session_dir"] = latest_session
                summary_path = latest_session / "summary.json"
                if summary_path.exists():
                    paths["summary"] = summary_path

        elif mode == "realtime":
            session_dir = self.output_realtime / video_name

            if session_dir.exists():
                paths["session_dir"] = session_dir

                # Find video
                video_candidates = [
                    session_dir / "session_h264.mp4",
                    session_dir / "session.mp4",
                ]
                for candidate in video_candidates:
                    if candidate.exists():
                        paths["video"] = candidate
                        break

                # Find realtime data
                realtime_data = session_dir / "realtime_emotions.json"
                if realtime_data.exists():
                    paths["summary"] = realtime_data

        return paths

    def validate_video_files(
        self, video_name: str, mode: str = "offline"
    ) -> Dict[str, bool]:
        """
        Check which files exist for a video.

        Args:
            video_name: Name of the video (without extension)
            mode: "offline" or "realtime"

        Returns:
            Dictionary with existence flags: has_video, has_summary, has_session
        """
        paths = self.get_video_paths(video_name, mode)

        return {
            "has_video": paths["video"] is not None and paths["video"].exists(),
            "has_summary": paths["summary"] is not None and paths["summary"].exists(),
            "has_session": paths["session_dir"] is not None
            and paths["session_dir"].exists(),
        }

    def get_all_sessions(
        self, video_name: str, mode: str = "offline"
    ) -> List[Tuple[str, Path]]:
        """
        Get all sessions for a video with their timestamps.

        Args:
            video_name: Name of the video
            mode: "offline" or "realtime"

        Returns:
            List of (session_name, session_path) tuples, sorted newest first
        """
        sessions = []

        if mode == "offline":
            base_dir = self.output_reports_offline / video_name / "frames_fps5"

            if base_dir.exists():
                for session_dir in base_dir.iterdir():
                    if session_dir.is_dir():
                        sessions.append((session_dir.name, session_dir))

                # Sort by name (timestamp) descending
                sessions.sort(key=lambda x: x[0], reverse=True)

        return sessions

    def resolve_video_file(self, video_name: str) -> Optional[Path]:
        """
        Quick helper to just get the video file path.

        Args:
            video_name: Name of the video

        Returns:
            Path to video file or None
        """
        paths = self.get_video_paths(video_name, "offline")
        return paths.get("video")

    def resolve_summary_file(
        self, video_name: str, mode: str = "offline"
    ) -> Optional[Path]:
        """
        Quick helper to just get the summary/data file path.

        Args:
            video_name: Name of the video
            mode: "offline" or "realtime"

        Returns:
            Path to summary file or None
        """
        paths = self.get_video_paths(video_name, mode)
        return paths.get("summary")
