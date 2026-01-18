from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query

from src.core.pipeline_executor import PipelineExecutor
from src.core.video_manager import VideoManager
from src.core.models import PipelineConfig
from app.dependencies import get_pipeline_executor, get_video_manager
from app.schemas import PipelineRunRequest, PipelineJobResponse

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])


@router.post("/run", response_model=PipelineJobResponse)
async def run_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor),
    video_manager: VideoManager = Depends(get_video_manager),
):
    """Start pipeline execution for a video"""
    # Create config from options
    options = request.options or {}
    config = PipelineConfig(
        fps=options.get("fps", 5),
        smoothing=options.get("smoothing", True),
        tta=options.get("tta", True),
        backend=options.get("backend", "hsemotion"),
        visualize=options.get("visualize", True),
        overwrite=options.get("overwrite", False),
        export_bboxes=options.get("export_bboxes", True),
        no_extract=options.get("no_extract", False),
        no_detect=options.get("no_detect", False),
        no_analyze=options.get("no_analyze", False),
        no_summary=options.get("no_summary", False),
        no_visualize=options.get("no_visualize", False),
    )

    # Create job
    try:
        job_id = pipeline_executor.create_job(request.video_name, config)
    except Exception as e:
        # Map logic errors to HTTP exceptions if needed, but create_job usually works or fails hard
        raise HTTPException(status_code=500, detail=str(e))

    # Define wrapper to update video manager after execution
    def run_pipeline_and_refresh(jid: str):
        pipeline_executor.execute_job(jid)
        # Re-scan to detect new output files and update status
        video_manager.scan_videos()

    # Start execution in background
    background_tasks.add_task(run_pipeline_and_refresh, job_id)

    job = pipeline_executor.get_job(job_id)

    return PipelineJobResponse(
        job_id=job.job_id,
        video_name=job.video_name,
        status=job.status.value,
        created_at=job.created_at.isoformat(),
        started_at=None,
        completed_at=None,
        progress=None,
        logs=[],
        error=None,
    )


@router.get("/jobs/{job_id}", response_model=PipelineJobResponse)
async def get_job_status(
    job_id: str, pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor)
):
    """Get pipeline job status"""
    job = pipeline_executor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return PipelineJobResponse(
        job_id=job.job_id,
        video_name=job.video_name,
        status=job.status.value,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress.to_dict() if job.progress else None,
        logs=job.logs[-50:],  # Last 50 log lines
        error=job.error,
    )


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(50, ge=1, le=100),
    pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor),
):
    """List recent pipeline jobs"""
    jobs = pipeline_executor.list_jobs(limit=limit)

    responses = []
    for job in jobs:
        responses.append(
            PipelineJobResponse(
                job_id=job.job_id,
                video_name=job.video_name,
                status=job.status.value,
                created_at=job.created_at.isoformat(),
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                progress=job.progress.to_dict() if job.progress else None,
                logs=job.logs[-10:],  # Last 10 log lines for list view
                error=job.error,
            )
        )

    return {"jobs": responses, "total": len(responses)}


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str, pipeline_executor: PipelineExecutor = Depends(get_pipeline_executor)
):
    """Cancel a running pipeline job"""
    success = pipeline_executor.cancel_job(job_id)

    if success:
        return {"success": True, "message": "Job cancelled"}
    else:
        raise HTTPException(
            status_code=400, detail="Job cannot be cancelled (not running or not found)"
        )
