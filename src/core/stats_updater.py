"""
Stats Updater - Handles automatic recalculation of statistics after video operations.
"""

import subprocess
import sys
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class StatsUpdater:
    """Manages statistics recalculation"""
    
    def __init__(self, project_root: Path, config_path: Optional[Path] = None):
        self.project_root = Path(project_root)
        self.config_path = config_path or (self.project_root / "config.yaml")
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def recalculate_all_stats(self) -> bool:
        """Regenerate all summary reports (offline + realtime)"""
        logger.info("Recalculating all statistics...")
        
        try:
            offline_ok = self._recalculate_offline_stats()
            realtime_ok = self._recalculate_realtime_stats()
            
            if offline_ok and realtime_ok:
                logger.info("Successfully recalculated all statistics")
                return True
            else:
                logger.warning("Some statistics failed to recalculate")
                return False
        
        except Exception as e:
            logger.error(f"Failed to recalculate statistics: {e}")
            return False
    
    async def recalculate_all_stats_async(self) -> bool:
        """Asynchronously regenerate all summary reports"""
        loop = asyncio.get_event_loop()
        
        offline_task = loop.run_in_executor(self._executor, self._recalculate_offline_stats)
        realtime_task = loop.run_in_executor(self._executor, self._recalculate_realtime_stats)
        
        offline_ok, realtime_ok = await asyncio.gather(offline_task, realtime_task)
        
        if offline_ok and realtime_ok:
            logger.info("Successfully recalculated all statistics (async)")
            return True
        else:
            logger.warning("Some statistics failed to recalculate (async)")
            return False
    
    def _recalculate_offline_stats(self) -> bool:
        """Regenerate offline summary reports"""
        script = self.project_root / "src" / "offline" / "emotion_summary_report.py"
        
        if not script.exists():
            logger.warning(f"Offline summary script not found: {script}")
            return False
        
        emotion_results_dir = self.project_root / "output" / "emotion_results"
        reports_dir = self.project_root / "output" / "reports"
        
        cmd = [
            sys.executable,
            str(script),
            "--project-root", str(self.project_root),
            "--config", str(self.config_path),
            "--input-dir", str(emotion_results_dir),
            "--output-dir", str(reports_dir),
        ]
        
        try:
            logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Offline statistics recalculated successfully")
                return True
            else:
                logger.error(f"Offline stats failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error("Offline stats recalculation timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to run offline stats: {e}")
            return False
    
    def _recalculate_realtime_stats(self) -> bool:
        """Regenerate realtime summary reports"""
        script = self.project_root / "src" / "realtime" / "summarize_master.py"
        
        if not script.exists():
            logger.debug(f"Realtime summary script not found: {script}")
            return True  # Not an error if realtime doesn't exist
        
        cmd = [sys.executable, str(script)]
        
        try:
            logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Realtime statistics recalculated successfully")
                return True
            else:
                logger.warning(f"Realtime stats warning: {result.stderr}")
                return True  # Don't fail if realtime has issues
        
        except subprocess.TimeoutExpired:
            logger.error("Realtime stats recalculation timed out")
            return False
        except Exception as e:
            logger.warning(f"Realtime stats error (non-critical): {e}")
            return True
    
    def recalculate_video_stats(self, video_name: str, mode: str = "offline") -> bool:
        """Regenerate stats for a specific video"""
        if mode == "offline":
            return self._recalculate_offline_video_stats(video_name)
        else:
            # Realtime stats are global, recalculate all
            return self._recalculate_realtime_stats()
    
    def _recalculate_offline_video_stats(self, video_name: str) -> bool:
        """Regenerate stats for a specific offline video"""
        script = self.project_root / "src" / "offline" / "emotion_summary_report.py"
        
        if not script.exists():
            logger.warning(f"Offline summary script not found: {script}")
            return False
        
        emotion_results_dir = self.project_root / "output" / "emotion_results"
        reports_dir = self.project_root / "output" / "reports"
        
        cmd = [
            sys.executable,
            str(script),
            "--project-root", str(self.project_root),
            "--config", str(self.config_path),
            "--input-dir", str(emotion_results_dir),
            "--output-dir", str(reports_dir),
            "--only-session", video_name,
        ]
        
        try:
            logger.debug(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Stats recalculated for {video_name}")
                return True
            else:
                logger.error(f"Stats failed for {video_name}: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to recalculate stats for {video_name}: {e}")
            return False
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
