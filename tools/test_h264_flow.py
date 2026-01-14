from pathlib import Path
import sys
import shutil

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.path_resolver import PathResolver
from core.video_manager import VideoManager

def verify_h264_flow():
    video_name = "test_h264_flow"
    videos_dir = project_root / "data" / "videos"
    video_dir = videos_dir / video_name
    video_path = video_dir / f"{video_name}.mp4"
    h264_path = video_dir / f"{video_name}_h264_unannotated.mp4"
    
    # Setup dummy video
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path.touch() # Dummy file
    h264_path.touch()  # Dummy H264
    
    print(f"Created dummy files in {video_dir}")
    
    # 1. Verify VideoManager filters it out
    metadata_path = project_root / "data" / "metadata.json"
    vm = VideoManager(project_root, metadata_path)
    videos = vm.scan_videos()
    
    found_main = False
    found_h264 = False
    
    for v in videos:
        if v.name == video_name:
            found_main = True
        if "_h264" in v.name:
            found_h264 = True
            
    if found_main and not found_h264:
        print("[OK] VideoManager correctly found main video and ignored H.264 duplicate.")
    else:
        print(f"[FAIL] VideoManager check failed. Main: {found_main}, H264: {found_h264}")

    # 2. Verify PathResolver prioritizes it
    resolver = PathResolver(project_root)
    paths = resolver.get_video_paths(video_name)
    resolved_video = paths.get("video")
    
    if resolved_video == h264_path:
        print("[OK] PathResolver correctly prioritized H.264 unannotated file.")
    else:
        print(f"[FAIL] PathResolver resolved to: {resolved_video}, expected: {h264_path}")

    # Cleanup
    shutil.rmtree(video_dir)
    print("Cleanup done.")

if __name__ == "__main__":
    verify_h264_flow()
