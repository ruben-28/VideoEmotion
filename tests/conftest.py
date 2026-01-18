import pytest
from pathlib import Path
from unittest.mock import MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.video_manager import VideoManager
from src.core.scanner import VideoScanner
from src.core.metadata import MetadataStore
from src.core.stats import StatsCalculator
from app.main import app
from fastapi.testclient import TestClient

@pytest.fixture
def mock_scanner():
    return MagicMock(spec=VideoScanner)

@pytest.fixture
def mock_store():
    return MagicMock(spec=MetadataStore)

@pytest.fixture
def mock_stats():
    return MagicMock(spec=StatsCalculator)

@pytest.fixture
def video_manager(tmp_path, mock_scanner, mock_store, mock_stats):
    vm = VideoManager(
        project_root=tmp_path,
        scanner=mock_scanner,
        store=mock_store,
        stats_calculator=mock_stats
    )
    return vm

@pytest.fixture
def client():
    return TestClient(app)
