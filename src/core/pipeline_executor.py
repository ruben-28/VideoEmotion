"""
Pipeline Executor - Manages pipeline job execution with async support.
"""

import subprocess
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid

from .models import (
    PipelineJob,
    PipelineConfig,
    PipelineProgress,
    JobStatus,
    VideoStatus,
)
from .pipeline_parser import PipelineLogParser

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Manages pipeline job execution"""

    def __init__(self, project_root: Path):
        """
        Initialize PipelineExecutor.

        Args:
            project_root (Path): Root directory of the project.
        """
        self.project_root = Path(project_root)
        self.jobs_file = self.project_root / "pipeline_jobs.json"
        self.jobs: Dict[str, PipelineJob] = {}
        self._load_jobs()
        self._executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent pipelines
        self._running_processes: Dict[str, subprocess.Popen] = {}
        self.parser = PipelineLogParser()

    def _load_jobs(self) -> None:
        """
        Load existing pipeline jobs from the persistent JSON file.

        Logic:
        1. Checks if the jobs file exists.
        2. Reads JSON data.
        3. Deserializes into PipelineJob objects.
        4. Populates self.jobs dictionary.

        Side Effects:
            - Modifies self.jobs.
            - Logs errors if loading fails.
        """
        if not self.jobs_file.exists():
            return

        try:
            with open(self.jobs_file, "r", encoding="utf-8") as f:
                jobs_data = json.load(f)

            for job_id, job_dict in jobs_data.items():
                self.jobs[job_id] = PipelineJob.from_dict(job_dict)

            logger.info(f"Loaded {len(self.jobs)} pipeline jobs")
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")

    def _save_jobs(self) -> None:
        """
        Save all current pipeline jobs to the persistent JSON file.

        Logic:
        1. Serializes all PipelineJob objects to dictionary format.
        2. Writes to pipeline_jobs.json.
        """
        try:
            jobs_data = {job_id: job.to_dict() for job_id, job in self.jobs.items()}
            with open(self.jobs_file, "w", encoding="utf-8") as f:
                json.dump(jobs_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def create_job(self, video_name: str, config: PipelineConfig) -> str:
        """
        Create and persist a new pipeline job.

        Args:
            video_name (str): Name of the video (including or excluding extension).
            config (PipelineConfig): Configuration for the pipeline run (fps, flags, etc.).

        Returns:
            str: The unique ID of the created job.
        """
        job_id = f"job_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

        job = PipelineJob(
            job_id=job_id,
            video_name=video_name,
            status=JobStatus.PENDING,
            config=config,
            created_at=datetime.now(),
        )

        self.jobs[job_id] = job
        self._save_jobs()

        logger.info(f"Created pipeline job: {job_id} for {video_name}")
        return job_id

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        """
        Retrieve a job by its ID.

        Args:
            job_id (str): ID of the job.

        Returns:
            Optional[PipelineJob]: The job object or None if not found.
        """
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[PipelineJob]:
        """
        List recent pipeline jobs, sorted by creation date (newest first).

        Args:
            limit (int): Maximum number of jobs to return.

        Returns:
            List[PipelineJob]: List of stored pipeline jobs.
        """
        jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running pipeline job.

        Logic:
        1. Checks if job exists and is running.
        2. Terminates the underlying subprocess if active.
        3. Updates job status to CANCELLED.
        4. Saves job state.

        Args:
            job_id (str): ID of the job to cancel.

        Returns:
            bool: True if cancellation was initiated/successful, False if job not found or not running.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status != JobStatus.RUNNING:
            return False

        # Kill process if running
        if job_id in self._running_processes:
            try:
                process = self._running_processes[job_id]
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Cancelled job: {job_id}")
            except Exception as e:
                logger.error(f"Failed to cancel job {job_id}: {e}")
                try:
                    process.kill()
                except:
                    pass
            finally:
                del self._running_processes[job_id]

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        self._save_jobs()

        return True

    async def execute_job_async(self, job_id: str) -> bool:
        """
        Execute pipeline job asynchronously using the internal thread pool.

        Args:
            job_id (str): ID of the job to execute.

        Returns:
            bool: Correlation of the thread submission (awaitable).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.execute_job, job_id)

    def _resolve_python_executable(self) -> str:
        """
        Find the best Python executable for running the pipeline subprocess.

        Logic:
        1. Checks for 'mp_env' (MediaPipe environment).
        2. Checks for 'venv' (Standard virtual environment).
        3. Fallback to current system executable (sys.executable).

        Returns:
            str: Path to the selected Python executable.
        """
        # 1. Try mp_env (preferred as it has working ML dependencies)
        mp_python = self.project_root / "mp_env" / "Scripts" / "python.exe"
        if mp_python.exists():
            return str(mp_python)

        # 2. Try generic venv
        venv_python = self.project_root / "venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)

        # 3. Fallback to current
        return sys.executable

    def execute_job(self, job_id: str) -> bool:
        """
        Execute pipeline job synchronously in a subprocess.

        Logic:
        1. Validates job existence and status.
        2. Updates job status to RUNNING.
        3. Resolves paths (pipeline script, video file, python executable).
        4. Constructs command line arguments based on job config.
        5. Launches the pipeline subprocess.
        6. Monitors stdout line-by-line:
           - Captures logs.
           - Parses progress using PipelineLogParser.
           - Updates job state/progress in real-time.
        7. Handles completion (success/error) and updates video global status.

        Args:
            job_id (str): ID of the job to run.

        Returns:
            bool: True if execution succeeded, False otherwise.
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False

        if job.status != JobStatus.PENDING:
            logger.warning(f"Job {job_id} is not pending (status: {job.status})")
            return False

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        job.logs = []
        self._save_jobs()

        # Update video metadata status to partial (processing)
        self._update_video_status(job.video_name, VideoStatus.PARTIAL)

        logger.info(f"Starting pipeline job: {job_id}")

        try:
            # Build command
            pipeline_script = self.project_root / "src" / "offline" / "pipeline.py"

            if not pipeline_script.exists():
                raise Exception(f"Pipeline script not found: {pipeline_script}")

            # Find video path
            video_path = self._find_video_path(job.video_name)
            if not video_path:
                raise Exception(f"Video not found: {job.video_name}")

            python_exe = self._resolve_python_executable()

            cmd = [
                python_exe,
                str(pipeline_script),
                "--video",
                str(video_path),
                "--project-root",
                str(self.project_root),
                "--fps",
                str(job.config.fps),
                "--py-detect",
                python_exe,
            ]

            # Add flags
            if job.config.no_extract:
                cmd.append("--no-extract")
            if job.config.no_detect:
                cmd.append("--no-detect")
            if job.config.no_analyze:
                cmd.append("--no-analyze")
            if job.config.no_summary:
                cmd.append("--no-summary")
            if job.config.no_visualize:
                cmd.append("--no-visualize")

            if job.config.export_bboxes:
                cmd.append("--export-bboxes")

            if job.config.overwrite:
                cmd.append("--force-visualize")

            # Execute
            logger.info("=" * 80)
            logger.info("PIPELINE EXECUTION DEBUG INFO:")
            logger.info(f"Project Root: {self.project_root}")
            logger.info(f"Working Directory (cwd): {self.project_root}")
            logger.info(f"Video Path: {video_path}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info("=" * 80)

            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            self._running_processes[job_id] = process

            # Read output line by line
            total_steps = 5
            current_step = 0

            for line in process.stdout:
                line = line.strip()
                if line:
                    job.logs.append(line)
                    logger.debug(f"[{job_id}] {line}")

                    # Update progress based on output
                    # prevent progress regression
                    new_step_info = self.parser.parse_line(line)
                    
                    if new_step_info:
                        idx, name, percent = new_step_info
                        if idx > current_step:
                            current_step = idx
                            job.progress = PipelineProgress(
                                current_step=name,
                                current_step_index=current_step,
                                total_steps=total_steps,
                                percent=percent,
                            )
                            self._save_jobs()

            # Wait for completion
            return_code = process.wait()

            if job_id in self._running_processes:
                del self._running_processes[job_id]

            if return_code == 0:
                job.status = JobStatus.DONE
                job.progress = PipelineProgress(
                    current_step="completed",
                    current_step_index=total_steps,
                    total_steps=total_steps,
                    percent=100.0,
                )
                job.completed_at = datetime.now()
                job.logs.append("✅ Pipeline completed successfully")

                # Update video metadata status to processed
                self._update_video_status(job.video_name, VideoStatus.PROCESSED)

                logger.info(f"Job {job_id} completed successfully")
                self._save_jobs()
                return True

            else:
                raise Exception(f"Pipeline exited with code {return_code}")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job.status = JobStatus.ERROR
            job.error = str(e)
            job.completed_at = datetime.now()
            job.logs.append(f"❌ Error: {e}")
            self._save_jobs()

            if job_id in self._running_processes:
                del self._running_processes[job_id]

            return False

    def _find_video_path(self, video_name: str) -> Optional[Path]:
        """
        Resolve absolute path to a video file.

        Logic:
        1. Checks if video_name already includes an extension and exists.
        2. Tries appending standard video extensions (.mp4, .avi, etc.).
        3. Tries matching the file stem if extensions are mixed.

        Args:
            video_name (str): Name or path of the video.

        Returns:
            Optional[Path]: Absolute path to the video or None if not found.
        """
        videos_dir = self.project_root / "data" / "videos"

        # Try with extension
        if Path(video_name).suffix:
            video_path = videos_dir / video_name
            if video_path.exists():
                return video_path

        # Try common extensions
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_path = videos_dir / f"{video_name}{ext}"
            if video_path.exists():
                return video_path

            # Try without extension if video_name already has one
            stem = Path(video_name).stem
            video_path = videos_dir / f"{stem}{ext}"
            if video_path.exists():
                return video_path

        return None

    def _update_video_status(self, video_name: str, status: VideoStatus) -> None:
        """
        Update the status of a video in the global metadata file.

        Args:
            video_name (str): Name of the video.
            status (VideoStatus): New status to apply (e.g., PARTIAL, PROCESSED).
        """
        try:
            # Load metadata
            metadata_path = self.project_root / "video_metadata.json"
            if not metadata_path.exists():
                logger.warning(f"Metadata file not found: {metadata_path}")
                return

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Find video by name
            video_id = None
            for vid, vdata in metadata.get("videos", {}).items():
                if (
                    vdata.get("name") == video_name
                    or vdata.get("name") == Path(video_name).stem
                ):
                    video_id = vid
                    break

            if not video_id:
                logger.warning(f"Video not found in metadata: {video_name}")
                return

            # Update status
            metadata["videos"][video_id]["status"] = status.value
            if status == VideoStatus.PROCESSED:
                metadata["videos"][video_id]["processed_at"] = (
                    datetime.now().isoformat()
                )

            # Save metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated video {video_name} status to {status.value}")

        except Exception as e:
            logger.error(f"Failed to update video status: {e}")

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Remove jobs older than a specified number of days to keep history clean.

        Args:
            days (int): Retention period in days.

        Returns:
            int: Number of jobs removed.
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        removed = 0

        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if job.created_at.timestamp() < cutoff:
                if job.status in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            removed += 1

        if removed > 0:
            self._save_jobs()
            logger.info(f"Cleaned up {removed} old jobs")

        return removed

    def __del__(self):
        """Cleanup on deletion"""
        # Terminate any running processes
        if hasattr(self, "_running_processes"):
            for job_id, process in list(self._running_processes.items()):
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                    except:
                        pass

        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
