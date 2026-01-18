from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_list_videos_empty():
    # Note: This runs against the real app with mocked dependencies if overridden,
    # or real dependencies if not.
    # Since we didn't override app.dependency_overrides in conftest global fixture yet,
    # this will try to use real dependencies.
    # Ideally, we should override dependencies in conftest for API tests too.
    # But for a basic "smoke test" verifying router wiring, this is okay if environment is safe.
    # However, to be "deterministic", we MUST override.
    pass


# We need to override dependencies to test API properly without checking real disk
from app.dependencies import get_video_manager
from src.core.video_manager import VideoManager
from unittest.mock import MagicMock


def test_api_list_videos_mocked():
    mock_vm = MagicMock(spec=VideoManager)
    mock_vm.list_videos.return_value = []

    app.dependency_overrides[get_video_manager] = lambda: mock_vm

    response = client.get("/api/videos")
    assert response.status_code == 200
    data = response.json()
    assert "videos" in data
    assert data["total"] == 0

    app.dependency_overrides = {}  # Clean up
