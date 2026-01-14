from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.path_resolver import PathResolver
from core.video_manager import VideoManager

def verify():
    resolver = PathResolver(project_root)
    metadata_path = project_root / "data" / "metadata.json"
    vm = VideoManager(project_root, metadata_path)
    
    print("Scanning videos...")
    videos = vm.scan_videos()
    print(f"Found {len(videos)} videos.")
    
    for v in videos:
        print(f"Checking {v.name}...")
        paths = resolver.get_video_paths(v.name)
        video_path = paths.get("video")
        
        if video_path and video_path.exists():
             print(f"  [OK] Found at: {video_path}")
             if v.name in str(video_path.parent.name) and v.name == video_path.stem:
                 print("  [OK] Is nested correctly.")
             else:
                 print("  [WARN] Not nested or mismatch?")
        else:
             print(f"  [FAIL] Video path resolution failed for {v.name}")

if __name__ == "__main__":
    verify()
