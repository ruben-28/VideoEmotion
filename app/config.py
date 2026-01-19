import yaml
from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# 1. Server Configuration (Environment Variables)
# -----------------------------------------------------------------------------
class ServerSettings(BaseSettings):
    """
    Server runtime settings loaded from environment variables or .env file.
    """

    HOST: str = Field("0.0.0.0", description="Host to bind the server to")
    PORT: int = Field(8000, description="Port to bind the server to")
    RELOAD: bool = Field(True, description="Enable auto-reload for development")
    ALLOWED_ORIGINS: List[str] = Field(
        ["*"], description="List of allowed CORS origins"
    )

    # Paths (optional overrides)
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1],
        description="Root directory of the project",
    )
    CONFIG_PATH: Path = Field(
        Path("config.yaml"), description="Path to the main config.yaml file"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


# -----------------------------------------------------------------------------
# 2. Project Configuration (YAML file)
# -----------------------------------------------------------------------------


class PathConfig(BaseModel):
    videos: Path = Path("data/videos")
    frames: Path = Path("data/extracted_frames")
    faces: Path = Path("data/detected_faces")
    features: Path = Path("data/features")
    metadata: Path = Path("video_metadata.json")
    visualizations: Path = Path("output/visualizations")
    reports: Path = Path("output/reports")
    realtime_output: Path = Path("output/realtime")
    emotion_results: Path = Path("output/emotion_results")
    trash: Path = Path("trash")


class ProjectConfig(BaseModel):
    name: str = "VideoEmotion"
    version: Union[str, float] = "1.0.0"
    paths: PathConfig = Field(default_factory=PathConfig)

    # Allow extra fields for other YAML sections (face_detection, emotion_analysis, etc.)
    model_config = {"extra": "allow"}


class Settings:
    """
    Combined settings provider.
    Access server settings via .server and project config via .project
    """

    def __init__(self):
        # Load server settings (env)
        self.server = ServerSettings()

        # Load project settings (yaml)
        config_path = self.server.PROJECT_ROOT / self.server.CONFIG_PATH
        self.project = self._load_yaml_config(config_path)

    def _load_yaml_config(self, path: Path) -> ProjectConfig:
        if not path.exists():
            # Fallback default if missing
            return ProjectConfig()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Extract 'project' key if present, otherwise assume root is project
            project_data = data.get("project", {})

            # If yaml has specific paths, ensure they are mapped correctly
            # Pydantic handles partial updates if we pass dict
            return ProjectConfig(**project_data)
        except Exception as e:
            print(f"[WARNING] Failed to load config.yaml: {e}. Using defaults.")
            return ProjectConfig()


# Singleton instance
settings = Settings()
