# src/pipeline.py
import argparse
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

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
        str((project_root / "src" / "detect_faces.py").resolve()),
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
    video_name: str,
) -> None:
    import emotion_summary_report as report_mod

    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            saved_argv[0],
            "--project-root", str(project_root),
            "--config", str(config_path),
            "--input-dir", str(emotion_out_root),
            "--output-dir", str(reports_out_root),
            "--only-session", str(video_name),
        ]
        report_mod.main()
    finally:
        sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(
        description="VideoEmotion - Unified pipeline runner (extract -> detect -> analyze -> report)"
    )
    parser.add_argument("--video", required=True, help="Chemin vers la vidéo (ex: data/videos/x.mp4)")
    parser.add_argument("--fps", type=int, default=None, help="FPS extraction (override config si fourni)")
    parser.add_argument("--py-detect", default=None, help="Chemin vers python.exe du venv mp_env (mediapipe).")
    parser.add_argument("--project-root", default=None, help="Racine du projet (défaut: auto).")
    parser.add_argument("--config", default=None, help="Chemin config.yaml (défaut: <project-root>/config.yaml).")

    # ✅ Modes
    parser.add_argument("--no-extract", action="store_true", help="Skip étape 1 (extract_frames).")
    parser.add_argument("--no-detect", action="store_true", help="Skip étape 2 (detect_faces).")
    parser.add_argument("--no-analyze", action="store_true", help="Skip étape 3 (analyze_emotion).")
    parser.add_argument("--no-summary", action="store_true", help="Skip étape 4 (emotion_summary_report).")
    parser.add_argument("--summary-only", action="store_true", help="Ne faire QUE le summary/report (étape 4).")

    # NEW: BBOX export via pipeline
    parser.add_argument("--export-bboxes", action="store_true", help="Demande à detect_faces d'exporter bboxes.json.")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    add_src_to_syspath(project_root)

    config_path = resolve_from_project(project_root, args.config) if args.config else (project_root / "config.yaml")
    cfg = load_config(config_path)

    # FPS: CLI > config > default
    cfg_fps = cfg_get(cfg, "frame_extraction", "fps", default=5)
    fps = args.fps if args.fps is not None else int(cfg_fps)
    fps = max(1, int(fps))

    # Paths: config > defaults
    videos_dir = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "videos", default="data/videos")))
    extracted_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "extracted_frames", default="data/extracted_frames")))
    detected_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "detected_faces", default="data/detected_faces")))
    emotion_out_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "emotion_results", default="output/emotion_results")))
    reports_out_root = resolve_from_project(project_root, str(cfg_get(cfg, "paths", "reports", default="output/reports")))

    # py_detect: CLI > config > None
    cfg_py_detect = cfg_get(cfg, "runtime", "py_detect", default=None)
    py_detect = args.py_detect or (str(cfg_py_detect) if cfg_py_detect else None)

    # video path
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (project_root / video_path).resolve()
    else:
        video_path = video_path.resolve()

    if not video_path.exists():
        # fallback: if user passed only filename, try videos_dir/<filename>
        alt = (videos_dir / Path(args.video).name).resolve()
        if alt.exists():
            video_path = alt
        else:
            eprint(f"[ERREUR] Vidéo introuvable: {video_path}")
            eprint(f"  (Testé aussi: {alt})")
            sys.exit(1)

    video_name = video_path.stem

    # Per-video dirs
    extracted_video_root = (extracted_root / video_name)
    detected_video_root = (detected_root / video_name)
    emotion_video_root = (emotion_out_root / video_name)

    # Master json global (toutes vidéos)
    master_json_path = (emotion_out_root / "emotion_results_master.json")

    # =========================================================
    # Summary-only mode
    # =========================================================
    if args.summary_only:
        print("\n==============================")
        print("[SUMMARY ONLY] emotion_summary_report")
        print("==============================")
        if not emotion_out_root.exists():
            eprint(f"[ERREUR] Dossier emotion_results introuvable: {emotion_out_root}")
            sys.exit(1)

        run_summary_for_video(
            project_root=project_root,
            config_path=config_path,
            emotion_out_root=emotion_out_root,
            reports_out_root=reports_out_root,
            video_name=video_name,
        )
        print("\n✅ Summary terminé.")
        return

    # =========================================================
    # 1) Extract frames
    # =========================================================
    if args.no_extract:
        print("\n[SKIP] --no-extract activé : étape 1 sautée.")
        if not extracted_video_root.exists():
            eprint(f"[ERREUR] --no-extract mais frames introuvables: {extracted_video_root}")
            sys.exit(1)
    else:
        print("\n==============================")
        print("[1/4] extract_frames")
        print("==============================")
        print(f"[CONFIG] fps={fps}")
        print(f"[CONFIG] extracted_video_root={extracted_video_root}")

        from extract_frames import extract_frames

        extracted_video_root.mkdir(parents=True, exist_ok=True)
        ok = extract_frames(
            video_path=str(video_path),
            output_folder=str(extracted_video_root),
            frame_rate=fps,
        )
        if not ok:
            eprint("[ERREUR] extract_frames a échoué.")
            sys.exit(1)

    # =========================================================
    # 2) Detect faces
    # =========================================================
    if args.no_detect:
        print("\n[SKIP] --no-detect activé : étape 2 sautée.")
        if not detected_video_root.exists():
            eprint(f"[ERREUR] --no-detect mais faces introuvables: {detected_video_root}")
            sys.exit(1)
    else:
        print("\n==============================")
        print("[2/4] detect_faces")
        print("==============================")
        print(f"[CONFIG] detected_video_root={detected_video_root}")
        print(f"[CONFIG] export_bboxes={bool(args.export_bboxes)}")

        if py_detect is None:
            eprint("[ERREUR] detect_faces nécessite --py-detect (python.exe du mp_env) ou runtime.py_detect dans config.yaml.")
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
            eprint(f"[ERREUR] detect_faces a échoué (code={rc}).")
            sys.exit(rc)

    # =========================================================
    # 3) Analyze emotions
    # =========================================================
    if args.no_analyze:
        print("\n[SKIP] --no-analyze activé : étape 3 sautée.")
        if not emotion_video_root.exists() and not master_json_path.exists():
            eprint(f"[ERREUR] --no-analyze mais résultats introuvables: {emotion_video_root} et {master_json_path}")
            sys.exit(1)
    else:
        print("\n==============================")
        print("[3/4] analyze_emotion")
        print("==============================")
        print(f"[CONFIG] emotion_video_root={emotion_video_root}")
        print(f"[CONFIG] master_json={master_json_path}")

        from analyze_emotion import analyze_emotions_incremental

        emotion_video_root.mkdir(parents=True, exist_ok=True)

        analyze_emotions_incremental(
            faces_root=str(detected_video_root),
            output_root=str(emotion_video_root),
            master_json_path=str(master_json_path),
        )

    # =========================================================
    # 4) Summary report
    # =========================================================
    if args.no_summary:
        print("\n[SKIP] --no-summary activé : étape 4 sautée.")
        print("\n✅ Pipeline terminé (sans summary).")
        return

    print("\n==============================")
    print("[4/4] emotion_summary_report")
    print("==============================")

    run_summary_for_video(
        project_root=project_root,
        config_path=config_path,
        emotion_out_root=emotion_out_root,
        reports_out_root=reports_out_root,
        video_name=video_name,
    )

    print("\n✅ Pipeline terminé.")


if __name__ == "__main__":
    main()
