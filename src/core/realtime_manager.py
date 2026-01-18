import subprocess
import sys
import threading
import logging
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RealtimeStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class RealtimeConfig(BaseModel):
    camera_id: int = 0
    display_width: int = 800
    min_det_score: float = 0.65
    save_json: bool = True
    save_video: bool = True
    visualize: bool = True


class RealtimeSession(BaseModel):
    session_id: str
    start_time: datetime
    status: RealtimeStatus
    config: RealtimeConfig
    output_dir: Optional[str] = None
    error: Optional[str] = None


class RealtimeManager:
    """
    Manages the realtime analysis subprocess.
    Ensures only one session runs at a time.
    """

    def __init__(self, project_root: Path, python_executable: str = sys.executable):
        self.project_root = project_root
        self.python_executable = python_executable
        self._process: Optional[subprocess.Popen] = None
        self._session: Optional[RealtimeSession] = None
        self._logs: List[str] = []
        self._log_lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_status(self) -> RealtimeSession:
        if self._session:
            # Update status based on process
            if self.is_running:
                self._session.status = RealtimeStatus.RUNNING
            elif self._session.status == RealtimeStatus.RUNNING:
                # Process exited unexpectedly or finished
                rc = self._process.poll()
                if rc == 0:
                    self._session.status = RealtimeStatus.IDLE
                else:
                    self._session.status = RealtimeStatus.ERROR
                    self._session.error = f"Process exited with code {rc}"
        else:
            # No active session object, so idle
            return RealtimeSession(
                session_id="none",
                start_time=datetime.now(),
                status=RealtimeStatus.IDLE,
                config=RealtimeConfig(),
            )
        return self._session

    def get_logs(self, limit: int = 100) -> List[str]:
        with self._log_lock:
            return list(self._logs[-limit:])

    def start_session(self, config: RealtimeConfig) -> RealtimeSession:
        if self.is_running:
            raise RuntimeError("A realtime session is already running")

        self._logs = []  # Clear logs
        self._stop_event.clear()

        script_path = self.project_root / "src" / "realtime" / "realtime_analysis.py"

        # Build command args
        cmd = [
            self.python_executable,
            str(script_path),
            "--camera-id",
            str(config.camera_id),
            "--project-root",
            str(self.project_root),
        ]

        if config.display_width > 0:
            cmd.extend(["--display-width", str(config.display_width)])

        cmd.extend(["--min-det-score", str(config.min_det_score)])

        if not config.save_json:
            cmd.append("--no-save-json")

        if not config.save_video:
            cmd.append("--no-save-video")

        if not config.visualize:
            cmd.append("--no-visualize")

        try:
            # Prepare environment with PYTHONPATH and encoding
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root)
            env["PYTHONIOENCODING"] = "utf-8"

            # Launch process
            # We use bufsize=1 for line buffered output
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr to stdout
                text=True,
                cwd=str(self.project_root),
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            # Start log consumer thread
            threading.Thread(target=self._consume_logs, daemon=True).start()

            self._session = RealtimeSession(
                session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                start_time=datetime.now(),
                status=RealtimeStatus.RUNNING,
                config=config,
                output_dir="output/realtime",  # Default
            )

            logger.info(f"Started realtime session {self._session.session_id}")
            return self._session

        except Exception as e:
            logger.error(f"Failed to start realtime session: {e}")
            self._session = None
            raise

    def stop_session(self) -> None:
        if not self.is_running:
            return

        logger.info("Stopping realtime session...")
        self._stop_event.set()

        if self._session:
            self._session.status = RealtimeStatus.STOPPING

        # Try graceful termination - relying on the script's ability to handle signals might be tricky on Windows
        # sending 'q' to stdin if it was checking that, but it checks cv2.waitKey
        # So we just terminate.

        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        except Exception as e:
            logger.error(f"Error stopping process: {e}")

        if self._session:
            self._session.status = RealtimeStatus.IDLE

    def _consume_logs(self):
        """Reads stdout from process and appends to logs"""
        if not self._process or not self._process.stdout:
            return

        try:
            for line in iter(self._process.stdout.readline, ""):
                line = line.strip()
                if line:
                    with self._log_lock:
                        self._logs.append(line)
                    # logger.info(f"[Realtime] {line}") # Optional: mirror to main log
        except Exception as e:
            logger.error(f"Log consumption error: {e}")
        finally:
            # Ensure process cleanup if it ends naturally
            self._process.stdout.close()
