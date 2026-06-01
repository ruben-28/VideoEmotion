# src/offline/utils.py
import json
from pathlib import Path
from typing import Dict, Any, Optional


def aggregate_video_results(emotion_video_root: Path, output_json_path: Path) -> Path:
    """
    Scan emotion_video_root for all analyzed_emotions.json files,
    merge them into a single dictionary, and save to output_json_path.

    Logic:
    - Recursively finds all 'analyzed_emotions.json'.
    - Merges content. If keys collide, prefixes them with the relative path to ensure uniqueness.
    - Used to aggregate potentially fragmented analysis results.

    Args:
        emotion_video_root (Path): Root directory to scan.
        output_json_path (Path): Path to save the aggregated JSON.

    Returns:
        Path: The path to the saved output file.
    """
    emotion_video_root = Path(emotion_video_root)
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Any] = {}
    files = sorted(emotion_video_root.rglob("analyzed_emotions.json"))

    if not files:
        raise FileNotFoundError(
            f"No analyzed_emotions.json found in: {emotion_video_root}"
        )

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Unable to read {fp}: {e}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] Unexpected format (not a dict) in {fp}")
            continue

        # Prefix = relative path of the file to make keys unique in case of collision
        prefix = fp.relative_to(emotion_video_root).as_posix()

        for k, v in data.items():
            nk = k
            if nk in merged:
                nk = f"{prefix}::{k}"
            merged[nk] = v

    output_json_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Aggregated JSON: {output_json_path} (from {len(files)} files)")
    return output_json_path


def frames_dir_name_from_fps(fps: int) -> str:
    return f"frames_fps{int(fps)}"


def resolve_source_video(videos_dir: Path, video_name: str) -> Optional[Path]:
    """
    Find source video file in videos_dir by trying standard extensions.

    Args:
        videos_dir (Path): Base directory for videos.
        video_name (str): Name of the video (stem).

    Returns:
        Optional[Path]: Absolute path to valid video file, or None.
    """
    videos_dir = Path(videos_dir)
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        p = videos_dir / (video_name + ext)
        if p.exists():
            return p
    return None


def maybe_find_bboxes_json(detected_video_root: Path, fps: int) -> Optional[Path]:
    """
    Attempt to locate bboxes.json in the expected path.
    Expected: data/detected_faces/<video_name>/frames_fpsX/bboxes.json

    Args:
        detected_video_root (Path): Root for detected faces of a specific video.
        fps (int): Frame rate used.

    Returns:
        Optional[Path]: Path to bboxes.json if it exists.
    """
    detected_video_root = Path(detected_video_root)
    p = detected_video_root / frames_dir_name_from_fps(fps) / "bboxes.json"
    return p if p.exists() else None


import subprocess
from typing import Tuple


def ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception:
        return False


def transcode_to_h264(src: Path, dst: Path) -> Tuple[bool, str]:
    """
    Transcode video to H.264 (yuv420p) for compatibility with web browsers.

    Args:
        src (Path): Source video path.
        dst (Path): Destination video path.

    Returns:
        Tuple[bool, str]: (Success status, Error message or stderr output).
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-movflags",
            "+faststart",
            "-an",
            str(dst),
        ]
        # Run without creating a window on Windows if possible, but subprocess.PIPE usually hides it.
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        ok = (proc.returncode == 0) and dst.exists()
        msg = proc.stderr[-1500:] if proc.stderr else ""
        return ok, msg
    except Exception as e:
        return False, str(e)
