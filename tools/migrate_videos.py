import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def migrate_videos():
    """
    Migrate videos from data/videos/*.mp4 to data/videos/<video_name>/<video_name>.mp4
    """
    project_root = Path(__file__).parent.parent
    videos_dir = project_root / "data" / "videos"
    
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return

    logger.info(f"Scanning {videos_dir}...")
    
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    count = 0
    
    for item in videos_dir.iterdir():
        if item.is_file() and item.suffix.lower() in video_extensions:
            video_name = item.stem
            target_dir = videos_dir / video_name
            target_path = target_dir / item.name
            
            # Skip if already in the right place (though iterating main dir shouldn't see files inside subdirs unless recursing)
            # Since we are iterating videos_dir, we are looking at files directly in data/videos
            
            logger.info(f"Migrating {item.name} -> {video_name}/{item.name}")
            
            try:
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(item), str(target_path))
                count += 1
            except Exception as e:
                logger.error(f"Failed to move {item.name}: {e}")

    logger.info(f"Migration completed. Moved {count} videos.")

if __name__ == "__main__":
    migrate_videos()
