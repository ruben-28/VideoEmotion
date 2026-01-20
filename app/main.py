from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

from app.config import settings
from app.routers import videos, trash, pipeline, stats, realtime
from app.dependencies import get_video_manager, get_pipeline_executor
from src.core.exceptions import AppError, VideoNotFoundError

# Config Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="VideoEmotion API",
    description="REST API for video emotion analysis administration",
    version=str(settings.project.version),
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler
@app.exception_handler(AppError)
async def app_exception_handler(request: Request, exc: AppError):
    """
    Handle known application errors (AppError).
    Maps specific exceptions (like VideoNotFoundError) to appropriate HTTP status codes.
    """
    status_code = 500
    if isinstance(exc, VideoNotFoundError):
        status_code = 404

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "path": request.url.path,
            }
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler for unexpected server errors.
    Logs the error stack trace and returns a generic 500 response.
    """
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred. Please check logs.",
                "details": "Internal Error",
            }
        },
    )


# Include Routers
app.include_router(videos.router)
app.include_router(trash.router)
app.include_router(pipeline.router)
app.include_router(stats.router)
app.include_router(realtime.router)

# Mount Static Files
# Note: "output" handles both reports and realtime.
app.mount(
    "/static/videos",
    StaticFiles(directory=settings.server.PROJECT_ROOT / settings.project.paths.videos),
    name="videos",
)
app.mount(
    "/static/output",
    StaticFiles(directory=settings.server.PROJECT_ROOT / "output"),
    name="output",
)


@app.get("/")
async def root():
    """
    Root endpoint to verify API availability and version.
    """
    return {
        "name": settings.project.name,
        "version": settings.project.version,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint for monitoring purposes.
    """
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.
    
    Tasks:
    1. Initialize singleton services (VideoManager, PipelineExecutor).
    2. Perform an initial video scan synchronization.
    3. Clean up old pipeline jobs (older than 7 days).
    """
    logger.info("VideoEmotion API starting up...")

    # Initialize Core Services (cached)
    video_manager = get_video_manager()
    pipeline_executor = get_pipeline_executor()

    # Scan videos on startup
    try:
        video_manager.scan_videos()
        logger.info("Initial video scan completed")
    except Exception as e:
        logger.error(f"Initial video scan failed: {e}")

    # Cleanup old jobs
    try:
        removed = pipeline_executor.cleanup_old_jobs(days=7)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old pipeline jobs")
    except Exception as e:
        logger.error(f"Job cleanup failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server.HOST,
        port=settings.server.PORT,
        reload=settings.server.RELOAD,
        log_level="info",
    )
