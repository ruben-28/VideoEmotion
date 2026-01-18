import cv2
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def cfg_get(cfg: Dict[str, Any], *keys, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_from_project(project_root: Path, p: Optional[str]) -> Path:
    if p is None:
        return project_root
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (project_root / pp).resolve()


def extract_frames(video_path, output_folder, frame_rate=5):
    """
    Extract frames into a FIXED folder (no timestamp):
    output_folder/frames_fps{frame_rate}/
    Example filename: frame_00023_t00012340.jpg
    """

    frames_dir = os.path.join(output_folder, f"frames_fps{frame_rate}")
    # SKIP si déjà traité
    if os.path.exists(frames_dir) and any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(frames_dir)
    ):
        print(f"[SKIP] Frames déjà extraites: {frames_dir}")
        return True

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    interval_ms = 1000.0 / frame_rate
    next_capture_time = 0.0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if current_time_ms >= next_capture_time:
            current_time_ms_int = int(current_time_ms)

            frame_filename = os.path.join(
                frames_dir,
                f"frame_{saved_frame_count:05d}_t{current_time_ms_int:08d}.jpg",
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            next_capture_time += interval_ms

    cap.release()
    print(f"Extracted {saved_frame_count} frames to {frames_dir}")
    return True


def extract_all_new_videos(videos_dir, extracted_root, frame_rate=5):
    os.makedirs(extracted_root, exist_ok=True)

    for filename in os.listdir(videos_dir):
        if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        video_path = os.path.join(videos_dir, filename)
        video_name = os.path.splitext(filename)[0]
        video_output_dir = os.path.join(extracted_root, video_name)

        os.makedirs(video_output_dir, exist_ok=True)

        print(f"[PROCESS] Extraction frames '{filename}' à {frame_rate} FPS...")
        extract_frames(video_path, video_output_dir, frame_rate=frame_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos (VideoEmotion)"
    )

    # CLI
    parser.add_argument(
        "--video",
        default=None,
        help="Vidéo spécifique (si fourni, on ne traite que celle-là).",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="Dossier contenant des vidéos (si --video non fourni).",
    )
    parser.add_argument(
        "--output", default=None, help="Dossier racine de sortie pour extracted_frames."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS extraction (override config si fourni).",
    )
    parser.add_argument(
        "--project-root", default=None, help="Racine du projet (défaut: auto)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Chemin vers config.yaml (défaut: <project-root>/config.yaml).",
    )

    args = parser.parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )
    config_path = (
        resolve_from_project(project_root, args.config)
        if args.config
        else (project_root / "config.yaml")
    )
    cfg = load_config(config_path)

    # Paths depuis config (si CLI absent)
    cfg_videos_dir = cfg_get(cfg, "paths", "videos", default="data/videos")
    cfg_extracted_root = cfg_get(
        cfg, "paths", "extracted_frames", default="data/extracted_frames"
    )

    videos_dir = (
        resolve_from_project(project_root, args.videos_dir)
        if args.videos_dir
        else resolve_from_project(project_root, str(cfg_videos_dir))
    )
    extracted_root = (
        resolve_from_project(project_root, args.output)
        if args.output
        else resolve_from_project(project_root, str(cfg_extracted_root))
    )

    # FPS: CLI > config > default
    cfg_fps = cfg_get(cfg, "frame_extraction", "fps", default=5)
    fps = args.fps if args.fps is not None else int(cfg_fps)
    fps = max(1, int(fps))

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = (project_root / video_path).resolve()
        else:
            video_path = video_path.resolve()

        # petit fallback: si user met juste "x.mp4"
        if not video_path.exists():
            alt = (videos_dir / args.video).resolve()
            if alt.exists():
                video_path = alt

        if not video_path.exists():
            print(f"[ERREUR] Vidéo introuvable: {video_path}")
            return

        video_name = video_path.stem
        video_output_dir = extracted_root / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[PROCESS] Extraction frames '{video_path.name}' à {fps} FPS...")
        extract_frames(str(video_path), str(video_output_dir), frame_rate=fps)
        return

    # Mode dossier
    if not videos_dir.exists():
        print(f"[ERREUR] Dossier videos introuvable: {videos_dir}")
        return

    extract_all_new_videos(str(videos_dir), str(extracted_root), frame_rate=fps)


if __name__ == "__main__":
    main()

# Exemple:
# python src/extract_frames.py --video data/videos/test_pipeline.mp4
# python src/extract_frames.py --fps 10
