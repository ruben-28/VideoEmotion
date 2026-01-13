
import subprocess
import sys
from pathlib import Path

# Mocking the PipelineExecutor logic
project_root = Path(r"C:\Users\ruben\Desktop\VideoEmotion")
video_name = "gemini2.mp4" # Assuming this file exists based on previous success in terminal

# Try to find the video
video_path = project_root / "data" / "videos" / video_name
if not video_path.exists():
    # Try finding with extension if not provided (though I provided it above)
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        candidate = project_root / "data" / "videos" / f"gemini2{ext}"
        if candidate.exists():
            video_path = candidate
            break

print(f"Video path: {video_path}")
print(f"Project root: {project_root}")

cmd = [
    r"C:\Users\ruben\Desktop\VideoEmotion\venv\Scripts\python.exe",
    str(project_root / "src" / "offline" / "pipeline.py"),
    "--video", str(video_path),
    "--project-root", str(project_root),
    "--fps", "5",
    # "--summary-only",
]

print(f"Running command: {cmd}")

process = subprocess.Popen(
    cmd,
    cwd=str(project_root),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

for line in process.stdout:
    print(line, end="")

return_code = process.wait()
print(f"Return code: {return_code}")
