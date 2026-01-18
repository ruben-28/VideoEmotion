class AppError(Exception):
    """Base exception for the application."""

    pass


class VideoNotFoundError(AppError):
    """Raised when a video is not found."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        super().__init__(f"Video with ID '{video_id}' not found.")


class PipelineError(AppError):
    """Raised when a pipeline job fails or cannot be started."""

    pass


class ConfigurationError(AppError):
    """Raised when configuration is invalid."""

    pass


class RealtimeSessionError(AppError):
    """Raised when a realtime session operation fails."""

    pass
