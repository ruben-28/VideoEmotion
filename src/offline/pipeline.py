# src/offline/pipeline.py
import argparse
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


def eprint(*args):
    print(*args, file=sys.stderr)


def add_src_to_syspath(project_root: Path) -> Path:
    src_dir = (project_root / "src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        eprint(f"[WARN] Impossible de lire config: {config_path} ({type(e).__name__}: {e})")
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


def run_detect_faces_cli_in_mp_env(
    py_detect: str,
    project_root: Path,
    config_path: Path,
    input_frames_root: Path,
    output_faces_root: Path,
    export_bboxes: bool,
) -> int:
    cmd = [
        py_detect,
        str((project_root / "src" / "offline" / "detect_faces.py").resolve()),
        "--project-root", str(project_root),
        "--config", str(config_path),
        "--input-frames", str(input_frames_root),
        "--output-faces", str(output_faces_root),
    ]
    if export_bboxes:
        cmd += ["--export-bboxes"]

    print(f"\n[PIPELINE] detect_faces via: {py_detect}")
    return subprocess.call(cmd, cwd=str(project_root))


def run_summary_for_video(
    project_root: Path,
    config_path: Path,
    emotion_out_root: Path,
    reports_out_root: Path,
    video_name: Optional[str],
) -> None:
    import offline.emotion_summary_report as report_mod

    saved_argv = sys.argv[:]
    try:
        new_argv = [
            saved_argv[0],
            "--project-root", str(project_root),
            "--config", str(config_path),
            "--input-dir", str(emotion_out_root),
            "--output-dir", str(reports_out_root),
        ]
        if video_name:
            new_argv += ["--only-session", str(video_name)]
        
        sys.argv = new_argv
        report_mod.main()
    finally:
        sys.argv = saved_argv


def run_pipeline_for_video(
    video_path: Path,
    project_root: Path,
    config_path: Path,
    cfg: Dict[str, Any],
    fps: int,
    args: argparse.Namespace,
) -> None:
    # Resolve roots from cfg/arguments again to be sure (or pass them)
    # Re-resolving locally to keep function pure-ish
    extracted_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "extracted_frames", default="data/extracted_frames")))
    detected_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "detected_faces", default="data/detected_faces")))
    emotion_out_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "emotion_results", default="output/emotion_results")))
    reports_out_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "reports", default="output/reports")))
    
    # py_detect
    cfg_py_detect = cfg_get(cfg, "runtime", "py_detect", default=None)
    py_detect = args.py_detect or (str(cfg_py_detect) if cfg_py_detect else None)

    video_name = video_path.stem
    print(f"\n###########################################################")
    print(f"### Processing Video: {video_name}")
    print(f"###########################################################")

    extracted_video_root = extracted_root / video_name
    detected_video_root = detected_root / video_name
    emotion_video_root = emotion_out_root / video_name

    master_json_path = (emotion_out_root / "emotion_results_master.json")

    # Summary-only logic
    if args.summary_only:
        print("[SUMMARY ONLY] Skipping extraction/detection/analysis.")
        if not emotion_out_root.exists():
            eprint(f"[ERROR] emotion_results dir not found: {emotion_out_root}")
            return # Skip this video
        
        run_summary_for_video(
            project_root=project_root,
            config_path=config_path,
            emotion_out_root=emotion_out_root,
            reports_out_root=reports_out_root,
            video_name=video_name,
        )
        print(f"✅ Summary done for {video_name}.")
        return

    # 1) Extract
    if args.no_extract:
        print("[SKIP] Step 1 (Extract)")
    else:
        print(f"[1/4] Extract frames (fps={fps})")
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
        print("[SKIP] Step 2 (Detect)")
    else:
        print(f"[2/4] Detect faces")
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
        print("[SKIP] Step 3 (Analyze)")
    else:
        print(f"[3/4] Analyze emotions")
        from offline.analyze_emotion import analyze_emotions_incremental
        emotion_video_root.mkdir(parents=True, exist_ok=True)
        analyze_emotions_incremental(
            faces_root=str(detected_video_root),
            output_root=str(emotion_video_root),
            master_json_path=str(master_json_path),
        )

    # 4) Summary
    if args.no_summary:
        print("[SKIP] Step 4 (Summary)")
    else:
        print(f"[4/4] Summary Report")
        run_summary_for_video(
            project_root=project_root,
            config_path=config_path,
            emotion_out_root=emotion_out_root,
            reports_out_root=reports_out_root,
            video_name=video_name,
        )
    
    print(f"✅ Pipeline finished for {video_name}.")


def main():
    parser = argparse.ArgumentParser(
        description="VideoEmotion - Offline pipeline runner (extract -> detect -> analyze -> report)"
    )
    # Mode selection: --video OR --all
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Chemin vers la vidéo (ex: data/videos/x.mp4)")
    group.add_argument("--all", action="store_true", help="Traiter TOUTES les vidéos trouvées dans data/videos (config).")

    parser.add_argument("--fps", type=int, default=None, help="FPS extraction (override config si fourni)")
    parser.add_argument("--py-detect", default=None, help="Chemin vers python.exe du venv mp_env (mediapipe).")
    parser.add_argument("--project-root", default=None, help="Racine du projet (défaut: auto).")
    parser.add_argument("--config", default=None, help="Chemin config.yaml (défaut: <project-root>/config.yaml).")

    # Pipeline Flags
    parser.add_argument("--no-extract", action="store_true", help="Skip étape 1 (extract_frames).")
    parser.add_argument("--no-detect", action="store_true", help="Skip étape 2 (detect_faces).")
    parser.add_argument("--no-analyze", action="store_true", help="Skip étape 3 (analyze_emotion).")
    parser.add_argument("--no-summary", action="store_true", help="Skip étape 4 (emotion_summary_report).")
    parser.add_argument("--summary-only", action="store_true", help="Ne faire QUE le summary/report (étape 4).")

    # BBOX export via pipeline
    parser.add_argument("--export-bboxes", action="store_true", help="Demande à detect_faces d'exporter bboxes.json.")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    add_src_to_syspath(project_root)

    config_path = resolve_from_project(project_root, args.config) if args.config else (project_root / "config.yaml")
    cfg = load_config(config_path)

    # FPS handling
    cfg_fps = cfg_get(cfg, "frame_extraction", "fps", default=5)
    fps = args.fps if args.fps is not None else int(cfg_fps)
    fps = max(1, int(fps))

    # Resolve videos directory for scan
    videos_dir = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "videos", default="data/videos")))

    # Collect videos
    videos_to_process: List[Path] = []

    if args.video:
        # Single video mode
        p = Path(args.video)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        
        if not p.exists():
            # Try finding in videos_dir
            alt = (videos_dir / Path(args.video).name).resolve()
            if alt.exists():
                p = alt
            else:
                eprint(f"[FATAL] Video not found: {args.video}")
                sys.exit(1)
        videos_to_process.append(p)
    
    elif args.all:
        # Batch mode
        if not videos_dir.exists():
            eprint(f"[FATAL] Video directory not found: {videos_dir}")
            sys.exit(1)
        
        # Scan extensions
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        for f in videos_dir.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                videos_to_process.append(f)
        
        if not videos_to_process:
            eprint(f"[WARN] No video files found in {videos_dir}")
            sys.exit(0)
        
        print(f"[BATCH] Found {len(videos_to_process)} videos in {videos_dir}")

    # Execution Loop
    for vid in videos_to_process:
        try:
            run_pipeline_for_video(
                video_path=vid,
                project_root=project_root,
                config_path=config_path,
                cfg=cfg,
                fps=fps,
                args=args
            )
        except Exception as e:
            eprint(f"[ERROR] Crash while processing {vid.name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate Global Summary (All Sessions Master)
    if args.all and not args.no_summary:
        print("\n###########################################################")
        print("### Batch Finished. Generating Global Summary (Master)...")
        print("###########################################################")
        run_summary_for_video(
            project_root=project_root,
            config_path=config_path,
            emotion_out_root=resolve_from_project(project_root, str(cfg_get(cfg, "paths", "emotion_results", default="output/emotion_results"))),
            reports_out_root=resolve_from_project(project_root, str(cfg_get(cfg, "paths", "reports", default="output/reports"))),
            video_name=None, # Global summary
        )


if __name__ == "__main__":
    main()
