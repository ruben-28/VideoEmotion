
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import yaml

# If pydantic-settings is not available, we use standard Pydantic + yaml load
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class ProjectConfig(BaseModel):
    name: str
    version: str or float
    description: str

class PathsConfig(BaseModel):
    videos: Path = Path("data/videos")
    extracted_frames: Path = Path("data/extracted_frames")
    detected_faces: Path = Path("data/detected_faces")
    emotion_results: Path = Path("output/emotion_results")
    reports: Path = Path("output/reports/offline")
    realtime_output: Path = Path("output/realtime")
    visualizations: Path = Path("output/visualizations")

class Settings(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    
    # Store other sections as dicts for now to avoid comprehensive mapping if not needed
    frame_extraction: Dict[str, Any] = {}
    face_detection: Dict[str, Any] = {}
    emotion_analysis: Dict[str, Any] = {}
    realtime: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    logging: Dict[str, Any] = {}
    
    @classmethod
    def load_from_yaml(cls, path: Path) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Global settings instance
import os
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

try:
    settings = Settings.load_from_yaml(CONFIG_PATH)
except Exception as e:
    print(f"WARNING: Failed to load config.yaml: {e}")
    # Fallback default
    settings = Settings(
        project=ProjectConfig(name="VideoEmotion", version="1.0", description="Default"),
        paths=PathsConfig()
    )
