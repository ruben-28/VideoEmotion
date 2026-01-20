# src/offline/pipeline.py
"""
Pipeline Orchestrator for Offline Video Analysis.

This script manages the end-to-end execution of the emotion analysis pipeline:
1. Frame Extraction (extract_frames.py)
2. Face Detection (detect_faces.py via MediaPipe)
3. Emotion Analysis (analyze_emotion.py via HSEmotion)
4. Reporting (emotion_summary_report.py)
5. Visualization (visualize_results.py)

It supports both single video processing and batch mode.
"""
import argparse
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("pipeline")


def eprint(*args):
    """Print to stderr."""
    logger.error(" ".join(map(str, args)))


def add_src_to_syspath(project_root: Path) -> Path:
    """Ensure src/ directory is in sys.path for imports."""
    src_dir = (project_root / "src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file safely."""
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        eprint(
            f"[WARN] Impossible de lire config: {config_path} ({type(e).__name__}: {e})"
        )
        return {}


def cfg_get(cfg: Dict[str, Any], *keys, default=None):
    """Deep retrieval of config keys with default value."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_from_project(project_root: Path, p: Optional[str]) -> Path:
    """Resolve a relative path against project root."""
    if p is None:
        return project_root
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (project_root / pp).resolve()


def run_detect_faces_cli_in_mp_env(
    py_detect: str,
    project_root: Path,
    config_path: Path,
    input_frames_root: Path,
    output_faces_root: Path,
    export_bboxes: bool,
) -> int:
    """
    Execute detect_faces.py logic, potentially in a separate Python environment (if py_detect configured).

    Args:
        py_detect: Path to Python executable (optional separate venv).
        project_root: Project root.
        config_path: Config file path.
        input_frames_root: Input directory (raw frames).
        output_faces_root: Output directory (cropped faces).
        export_bboxes: If True, saves bounding boxes to JSON.
    
    Returns:
        int: Return code from subprocess.
    """
    cmd = [
        py_detect,
        str((project_root / "src" / "offline" / "detect_faces.py").resolve()),
        "--project-root",
        str(project_root),
        "--config",
        str(config_path),
        "--input-frames",
        str(input_frames_root),
        "--output-faces",
        str(output_faces_root),
    ]
    if export_bboxes:
        cmd += ["--export-bboxes"]

    logger.info(f"\n[PIPELINE] detect_faces via: {py_detect}")
    return subprocess.call(cmd, cwd=str(project_root))


def run_summary_for_video(
    project_root: Path,
    config_path: Path,
    emotion_out_root: Path,
    reports_out_root: Path,
    video_name: Optional[str],
) -> None:
    """
    Run the reporting module to generate JSON/CSV summaries.

    Args:
        project_root: Root path.
        config_path: Config file.
        emotion_out_root: Source directory for emotion results.
        reports_out_root: Destination directory for reports.
        video_name: Optional specific video to summarize (None = all).
    """
    import offline.emotion_summary_report as report_mod

    saved_argv = sys.argv[:]
    try:
        new_argv = [
            saved_argv[0],
            "--project-root",
            str(project_root),
            "--config",
            str(config_path),
            "--input-dir",
            str(emotion_out_root),
            "--output-dir",
            str(reports_out_root),
        ]
        if video_name:
            new_argv += ["--only-session", str(video_name)]

        sys.argv = new_argv
        report_mod.main()
    finally:
        sys.argv = saved_argv


def run_visualize_for_video(
    project_root: Path,
    video_path: Path,
    videos_dir: Path,
    detected_video_root: Path,
    emotion_video_root: Path,
    visualizations_root: Path,
    video_name: str,
    fps: int,
    force: bool,
) -> None:
    """
    Execute step 5: Visualization.
    
    Features:
    - Aggregates frame-by-frame JSON results.
    - Generates MP4 overlaying emotion data on original video.
    
    Args:
        project_root: Project root.
        video_path: Path to source video file.
        videos_dir: Base videos directory.
        detected_video_root: Path containing bboxes.json.
        emotion_video_root: Path containing emotion JSONs per frame.
        visualizations_root: Output directory.
        video_name: Identifier of video.
        fps: Frame extraction rate used.
        force: If True, overwrite existing outputs.
    """
    from offline.utils import (
        aggregate_video_results,
        frames_dir_name_from_fps,
        resolve_source_video,
        maybe_find_bboxes_json,
    )

    frames_dir = frames_dir_name_from_fps(fps)

    # Prefer explicit video_path else search in data/videos
    src_video = (
        video_path
        if video_path.exists()
        else resolve_source_video(videos_dir, video_name)
    )
    if src_video is None or not src_video.exists():
        eprint(f"[WARN] Source video introuvable pour visualisation: {video_name}")
        return

    visualizations_root.mkdir(parents=True, exist_ok=True)

    # Per-video output dir
    per_video_dir = visualizations_root / video_name
    per_video_dir.mkdir(parents=True, exist_ok=True)

    # Expected outputs inside the per-video folder
    out_video = per_video_dir / f"{video_name}_annotated_raw.mp4"
    out_h264 = per_video_dir / f"{video_name}_annotated_h264.mp4"

    if not force and (out_h264.exists() or out_video.exists()):
        logger.info(
            f"[SKIP] Visualize already exists: {out_h264 if out_h264.exists() else out_video}"
        )
        return

    # Aggregate JSON -> INSIDE per-video folder
    emotion_scan_root = (
        emotion_video_root  # IMPORTANT: now already points to .../<video>/frames_fpsX
    )
    aggregated_json = per_video_dir / f"{video_name}_{frames_dir}_video_emotions.json"
    aggregate_video_results(emotion_scan_root, aggregated_json)

    # bbox json expected: detected_video_root/bboxes.json OR detected_video_root/.../bboxes.json (depending on your detect script)
    bbox_json = maybe_find_bboxes_json(detected_video_root, fps)

    script_path = (project_root / "src" / "offline" / "visualize_results.py").resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--video",
        str(src_video),
        "--results",
        str(aggregated_json),
        "--out",
        str(visualizations_root),
        "--use-smoothed",
        "--show-confidence",
        "--show-backend",
        "--bbox-fps",
        str(fps),
        "--copy-jsons",
    ]
    if bbox_json is not None:
        cmd += ["--bboxes-json", str(bbox_json)]

    logger.info("[5/5] Visualize")
    logger.info("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=False)

    if out_h264.exists():
        logger.info(f"[OK] Chrome-friendly: {out_h264}")
    elif out_video.exists():
        logger.info(f"[OK] RAW video: {out_video}")
        logger.warning(
            "No _h264.mp4 version found (ffmpeg might be missing or failed)."
        )
    else:
        logger.warning(
            "No output video found in subfolder (check visualize_results.py logs)."
        )


def run_pipeline_for_video(
    video_path: Path,
    project_root: Path,
    config_path: Path,
    cfg: Dict[str, Any],
    fps: int,
    args: argparse.Namespace,
    videos_dir: Path,
) -> None:
    """
    Main Logic Loop for a single video.

    Executes steps sequentially:
    1. Extract frames (FFmpeg via python).
    2. Detect faces (MediaPipe).
    3. Analyze emotions (HSEmotion).
    4. Generate Summary.
    5. Create Visualization.

    Skips steps if instructed by command line args (e.g. --no-detect).
    Handles directory creation and resolving paths for each step.
    """
    extracted_root = resolve_from_project(
        project_root,
        str(cfg_get(cfg, "paths", "extracted_frames", default="data/extracted_frames")),
    )
    detected_root = resolve_from_project(
        project_root,
        str(cfg_get(cfg, "paths", "detected_faces", default="data/detected_faces")),
    )
    emotion_out_root = resolve_from_project(
        project_root,
        str(cfg_get(cfg, "paths", "emotion_results", default="output/emotion_results")),
    )
    reports_out_root = resolve_from_project(
        project_root, str(cfg_get(cfg, "paths", "reports", default="output/reports"))
    )
    visualizations_root = resolve_from_project(
        project_root,
        str(cfg_get(cfg, "paths", "visualizations", default="output/visualizations")),
    )

    # ✅ IMPORTANT: if reports path is generic, keep consistent with your summary script default
    if str(reports_out_root).replace("\\", "/").rstrip("/") == "output/reports":
        reports_out_root = (project_root / "output" / "reports" / "offline").resolve()

    cfg_py_detect = cfg_get(cfg, "runtime", "py_detect", default=None)
    py_detect = args.py_detect or (str(cfg_py_detect) if cfg_py_detect else None)

    from offline.utils import frames_dir_name_from_fps

    frames_dir = frames_dir_name_from_fps(fps)  # ex: frames_fps5

    video_name = video_path.stem
    logger.info("\n###########################################################")
    logger.info(f"### Processing Video: {video_name}")
    logger.info("###########################################################")

    # ✅ New consistent structure:
    # extracted: data/extracted_frames/<video>/frames_fpsX/
    # detected : data/detected_faces/<video>/frames_fpsX/
    # emotion  : output/emotion_results/<video>/frames_fpsX/
    # Note: resolved paths make creating these safe.
    extracted_video_root = extracted_root / video_name / frames_dir
    detected_video_root = detected_root / video_name / frames_dir
    emotion_video_root = emotion_out_root / video_name / frames_dir

    master_json_path = emotion_out_root / "emotion_results_master.json"

    # Visualize-only logic
    if args.visualize_only:
        logger.info("[VISUALIZE ONLY] Running aggregation + visualization only.")
        run_visualize_for_video(
            project_root=project_root,
            video_path=video_path,
            videos_dir=videos_dir,
            detected_video_root=detected_video_root,
            emotion_video_root=emotion_video_root,
            visualizations_root=visualizations_root,
            video_name=video_name,
            fps=fps,
            force=args.force_visualize,
        )
        return

    # Summary-only logic
    if args.summary_only:
        logger.info("[SUMMARY ONLY] Skipping extraction/detection/analysis.")
        if not emotion_out_root.exists():
            eprint(f"[ERROR] emotion_results dir not found: {emotion_out_root}")
            return

        logger.info("[4/5] Summary Report")
        run_summary_for_video(
            project_root=project_root,
            config_path=config_path,
            emotion_out_root=emotion_out_root,
            reports_out_root=reports_out_root,
            video_name=video_name,
        )
        logger.info(f"Summary done for {video_name}.")
        return

    # 1) Extract
    if args.no_extract:
        logger.info("[SKIP] Step 1 (Extract)")
    else:
        logger.info(f"[1/5] Extract frames (fps={fps})")
        from offline.extract_frames import extract_frames

        extracted_video_root.mkdir(parents=True, exist_ok=True)
        ok = extract_frames(
            video_path=str(video_path),
            output_folder=str(extracted_video_root),
            frame_rate=fps,
        )
        if not ok:
            eprint(f"[ERROR] Extraction failed for {video_name}")
            sys.exit(1)

    # 2) Detect
    if args.no_detect:
        logger.info("[SKIP] Step 2 (Detect)")
    else:
        logger.info("[2/5] Detect faces")
        if py_detect is None:
            eprint("[ERROR] py-detect not configured.")
            sys.exit(1)

    detected_video_root.mkdir(parents=True, exist_ok=True)
    rc = run_detect_faces_cli_in_mp_env(
        py_detect=py_detect,
        project_root=project_root,
        config_path=config_path,
        input_frames_root=extracted_video_root,
        output_faces_root=detected_video_root,
        export_bboxes=bool(args.export_bboxes),
    )
    if rc != 0:
        eprint(f"[ERROR] Detection failed for {video_name}")
        sys.exit(rc)

    # 3) Analyze
    if args.no_analyze:
        logger.info("[SKIP] Step 3 (Analyze)")
    else:
        logger.info("[3/5] Analyze emotions")
        from offline.analyze_emotion import analyze_emotions_incremental

        emotion_video_root.mkdir(parents=True, exist_ok=True)
        analyze_emotions_incremental(
            faces_root=str(detected_video_root),
            output_root=str(emotion_video_root),
            master_json_path=str(master_json_path),
        )

    # 4) Summary
    if args.no_summary:
        logger.info("[SKIP] Step 4 (Summary)")
    else:
        logger.info("[4/5] Summary Report")
        run_summary_for_video(
            project_root=project_root,
            config_path=config_path,
            emotion_out_root=emotion_out_root,
            reports_out_root=reports_out_root,
            video_name=video_name,
        )

    # 5) Visualize
    if args.no_visualize:
        logger.info("[SKIP] Step 5 (Visualize)")
    else:
        run_visualize_for_video(
            project_root=project_root,
            video_path=video_path,
            videos_dir=videos_dir,
            detected_video_root=detected_video_root,
            emotion_video_root=emotion_video_root,
            visualizations_root=visualizations_root,
            video_name=video_name,
            fps=fps,
            force=args.force_visualize,
        )

    logger.info(f"✅ Pipeline finished for {video_name}.")


def main():
    """
    Command Line Interface Entry Point.
    Parses arguments and configures the pipeline run.
    """
    parser = argparse.ArgumentParser(
        description="VideoEmotion - Offline pipeline runner (extract -> detect -> analyze -> report -> visualize)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Chemin vers la vidéo (ex: data/videos/x.mp4)")
    group.add_argument(
        "--all",
        action="store_true",
        help="Traiter TOUTES les vidéos trouvées dans data/videos (config).",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS extraction (override config si fourni)",
    )
    parser.add_argument(
        "--py-detect",
        default=None,
        help="Chemin vers python.exe du venv mp_env (mediapipe).",
    )
    parser.add_argument(
        "--project-root", default=None, help="Racine du projet (défaut: auto)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Chemin config.yaml (défaut: <project-root>/config.yaml).",
    )

    parser.add_argument(
        "--no-extract", action="store_true", help="Skip étape 1 (extract_frames)."
    )
    parser.add_argument(
        "--no-detect", action="store_true", help="Skip étape 2 (detect_faces)."
    )
    parser.add_argument(
        "--no-analyze", action="store_true", help="Skip étape 3 (analyze_emotion)."
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip étape 4 (emotion_summary_report).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Ne faire QUE le summary/report (étape 4).",
    )

    parser.add_argument(
        "--no-visualize", action="store_true", help="Skip étape 5 (visualize_results)."
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Ne faire QUE l'agrégation + visualisation (étape 5).",
    )

    parser.add_argument(
        "--force-visualize",
        action="store_true",
        help="Recrée la vidéo annotée même si elle existe déjà.",
    )
    parser.add_argument(
        "--export-bboxes",
        action="store_true",
        help="Demande à detect_faces d'exporter bboxes.json.",
    )

    args = parser.parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[2]
    )
    add_src_to_syspath(project_root)

    config_path = (
        resolve_from_project(project_root, args.config)
        if args.config
        else (project_root / "config.yaml")
    )
    cfg = load_config(config_path)

    cfg_fps = cfg_get(cfg, "frame_extraction", "fps", default=5)
    fps = args.fps if args.fps is not None else int(cfg_fps)
    fps = max(1, int(fps))

    videos_dir = resolve_from_project(
        project_root, str(cfg_get(cfg, "paths", "videos", default="data/videos"))
    )

    videos_to_process: List[Path] = []

    if args.video:
        p = Path(args.video)
        if not p.is_absolute():
            p = (project_root / p).resolve()

        print(f"[DEBUG] Project path: {project_root}")
        print(f"[DEBUG] Video path resolve attempt 1: {p}")
        print(f"[DEBUG] Exists? {p.exists()}")

        if not p.exists():
            # Check nested structure first: data/videos/name/name.mp4
            alt_nested = (
                videos_dir / Path(args.video).stem / Path(args.video).name
            ).resolve()
            if alt_nested.exists():
                p = alt_nested
            else:
                # Check flat structure
                alt = (videos_dir / Path(args.video).name).resolve()
                if alt.exists():
                    p = alt

        if not p.exists():
            eprint(f"[FATAL] Video not found: {args.video} (tried: {p})")
            sys.exit(1)
        videos_to_process.append(p)

    elif args.all:
        if not videos_dir.exists():
            eprint(f"[FATAL] Video directory not found: {videos_dir}")
            sys.exit(1)

        exts = {".mp4", ".avi", ".mov", ".mkv"}

        # Scan nested folders
        for item in videos_dir.iterdir():
            if item.is_dir():
                # Look for video file with same name inside
                # Or any valid extension
                found = False
                for ext in exts:
                    cand = item / f"{item.name}{ext}"
                    if cand.exists():
                        videos_to_process.append(cand)
                        found = True
                        break

        # Scan flat files (legacy)
        for f in videos_dir.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                videos_to_process.append(f)

        if not videos_to_process:
            eprint(f"[WARN] No video files found in {videos_dir}")
            sys.exit(0)

        print(f"[BATCH] Found {len(videos_to_process)} videos in {videos_dir}")

    for vid in videos_to_process:
        try:
            run_pipeline_for_video(
                video_path=vid,
                project_root=project_root,
                config_path=config_path,
                cfg=cfg,
                fps=fps,
                args=args,
                videos_dir=videos_dir,
            )
        except Exception as e:
            eprint(f"[ERROR] Crash while processing {vid.name}: {e}")
            import traceback

            traceback.print_exc()

    if args.all and not args.no_summary:
        print("\n###########################################################")
        print("### Batch Finished. Generating Global Summary (Master)...")
        print("###########################################################")
        run_summary_for_video(
            project_root=project_root,
            config_path=config_path,
            emotion_out_root=resolve_from_project(
                project_root,
                str(
                    cfg_get(
                        cfg,
                        "paths",
                        "emotion_results",
                        default="output/emotion_results",
                    )
                ),
            ),
            reports_out_root=resolve_from_project(
                project_root,
                str(cfg_get(cfg, "paths", "reports", default="output/reports")),
            ),
            video_name=None,
        )


if __name__ == "__main__":
    main()
