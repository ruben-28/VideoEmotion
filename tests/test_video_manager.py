import pytest
from src.core.models import VideoMetadata, VideoMode, VideoStatus


def test_scan_videos_sync(
    video_manager, mock_scanner, mock_store, mock_stats, tmp_path
):
    # Setup real files to avoid stat() errors
    video_file = tmp_path / "video1.mp4"
    video_file.touch()

    # Configure Scanner Mock
    mock_scanner.scan_offline.return_value = [video_file]
    mock_scanner.scan_realtime.return_value = []

    # Configure Stats Mock
    mock_stats.load_offline_stats.return_value = {}

    # Execute
    videos = video_manager.scan_videos()

    # Verify
    assert len(videos) == 1
    assert videos[0].name == "video1"
    assert videos[0].mode == VideoMode.OFFLINE

    # Verify store interaction
    mock_store.set_video.assert_called()
    mock_store.save.assert_called()


@pytest.mark.asyncio
async def test_scan_videos_async(
    video_manager, mock_scanner, mock_store, mock_stats, tmp_path
):
    # Setup real session dir
    session_dir = tmp_path / "session_1"
    session_dir.mkdir()
    (session_dir / "realtime_emotions.json").touch()

    # Configure Scanner Mock
    mock_scanner.scan_offline.return_value = []
    mock_scanner.scan_realtime.return_value = [session_dir]
    mock_stats.calculate_realtime_stats.return_value = {}

    # Execute
    videos = await video_manager.scan_videos_async()

    # Verify
    assert len(videos) == 1
    assert videos[0].name == "session_1"
    assert videos[0].mode == VideoMode.REALTIME

    # Verify store interaction
    mock_store.set_video.assert_called()
    mock_store.save.assert_called()


from datetime import datetime


def test_list_videos_filtering(video_manager, mock_store):
    # Setup mock data
    now = datetime.now()
    v1 = VideoMetadata(
        id="v1",
        name="v1",
        mode=VideoMode.OFFLINE,
        status=VideoStatus.PROCESSED,
        created_at=now,
        file_paths={},
        file_size_bytes=0,
        stats={},
    )
    v2 = VideoMetadata(
        id="v2",
        name="v2",
        mode=VideoMode.REALTIME,
        status=VideoStatus.UNPROCESSED,
        created_at=now,
        file_paths={},
        file_size_bytes=0,
        stats={},
    )

    # Mock list_videos returns dict of {id: dict}
    mock_store.list_videos.return_value = {"v1": v1.to_dict(), "v2": v2.to_dict()}

    # Test Filter by Mode
    offline = video_manager.list_videos(mode=VideoMode.OFFLINE)
    assert len(offline) == 1
    assert offline[0].id == "v1"

    # Test Filter by Status
    unprocessed = video_manager.list_videos(status=VideoStatus.UNPROCESSED)
    assert len(unprocessed) == 1
    assert unprocessed[0].id == "v2"
