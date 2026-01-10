# src/realtime/realtime_analysis.py
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import mediapipe as mp
import yaml

import json

from src.core.emotion.emotion_infer import EmotionInfer


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


def format_time_from_ms(ms: int) -> str:
    """Format ms -> HH:MM:SS.mmm"""
    if ms is None or ms < 0:
        return "00:00:00.000"
    s = ms // 1000
    mmm = ms % 1000
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"


def clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return x, y, w, h


def pick_largest_detection(detections, W: int, H: int):
    best = None
    best_area = 0
    for det in detections:
        rb = det.location_data.relative_bounding_box
        x = int(rb.xmin * W)
        y = int(rb.ymin * H)
        w = int(rb.width * W)
        h = int(rb.height * H)
        x, y, w, h = clip_box(x, y, w, h, W, H)
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


def main():
    ap = argparse.ArgumentParser(description="Realtime emotion analysis (HSEmotion only, largest face).")
    ap.add_argument("--camera-id", type=int, default=0, help="Webcam index (default 0).")
    ap.add_argument("--project-root", default=None, help="Project root (default auto).")
    ap.add_argument("--config", default=None, help="config.yaml path (default <project-root>/config.yaml).")

    # CLI overrides (optional)
    ap.add_argument("--display-width", type=int, default=None, help="Resize width for display (override config).")
    ap.add_argument("--min-det-score", type=float, default=None, help="Override min detection score (mediapipe).")
    ap.add_argument("--bbox-thickness", type=int, default=3, help="Épaisseur bbox.")
    ap.add_argument("--font-scale", type=float, default=0.9, help="Taille du texte.")
    ap.add_argument("--text-thickness", type=int, default=2, help="Épaisseur du texte.")

    # Save settings (Default: ON)
    ap.add_argument("--no-save-json", action="store_true", help="Désactiver la sauvegarde JSON.")
    ap.add_argument("--no-save-video", action="store_true", help="Désactiver l'enregistrement vidéo.")
    ap.add_argument("--out-dir", default="output/realtime", help="Dossier de sortie (JSON/vidéo).")

    args = ap.parse_args()

    # Determine saving behavior
    do_save_json = not args.no_save_json
    do_save_video = not args.no_save_video

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    config_path = (
        (project_root / "config.yaml")
        if not args.config
        else (Path(args.config).resolve() if Path(args.config).is_absolute() else (project_root / args.config).resolve())
    )
    cfg = load_config(config_path)

    # -------------------------------
    # Realtime config (with fallback)
    # -------------------------------
    # mirror
    mirror_fix = bool(cfg_get(cfg, "realtime", "mirror_fix", default=True))

    # display width: CLI > config > 0
    cfg_display_width = int(cfg_get(cfg, "realtime", "display_width", default=0) or 0)
    display_width = args.display_width if args.display_width is not None else cfg_display_width

    # emotion thresholds (realtime > fallback emotion_analysis)
    hse_thr = float(cfg_get(cfg, "realtime", "emotion", "hse_conf_threshold",
                            default=cfg_get(cfg, "emotion_analysis", "hsemotion", "confidence_threshold", default=0.65)))
    enable_uncertain = bool(cfg_get(cfg, "realtime", "emotion", "enable_uncertain",
                                    default=cfg_get(cfg, "emotion_analysis", "uncertain", "enabled", default=True)))
    uncertain_min = float(cfg_get(cfg, "realtime", "emotion", "uncertain_min_conf",
                                  default=cfg_get(cfg, "emotion_analysis", "uncertain", "min_conf", default=0.55)))

    # detection threshold: CLI > realtime > offline face_detection.filters
    if args.min_det_score is not None:
        min_det_score = float(args.min_det_score)
    else:
        min_det_score = float(cfg_get(cfg, "realtime", "face_detection", "min_det_score",
                                      default=cfg_get(cfg, "face_detection", "filters", "min_det_score", default=0.70)))

    infer = EmotionInfer(
        device="cpu",
        hse_threshold=hse_thr,
        enable_uncertain=enable_uncertain,
        uncertain_min_conf=uncertain_min,
    )

    # -------------------------------
    # Output session setup
    # -------------------------------
    session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()
    
    session_dir = out_root / f"session_{session_ts}"
    
    # Create folder only if we are saving something
    if do_save_json or do_save_video:
        out_root.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=True)

        json_path = session_dir / "realtime_emotions.json"
        video_path = session_dir / "session.mp4"

    records = []
    video_writer = None

    # Track session start time (ms) for readable timestamps
    session_start_ms = None

    # -------------------------------
    # Camera
    # -------------------------------
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la caméra id={args.camera_id}")
        return

    # Init video writer (if requested)
    if do_save_video:
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        if not fps_out or fps_out <= 1:
            fps_out = 25.0
        W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # video_path ne peut être None si do_save_video True
        video_writer = cv2.VideoWriter(str(video_path), fourcc, float(fps_out), (W0, H0))
        if not video_writer.isOpened():
            print("[WARN] Impossible d'ouvrir VideoWriter, désactivation save-video.")
            video_writer = None
            do_save_video = False  # important: pour cohérence fin de script

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_det_score) as face_det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror_fix:
                frame = cv2.flip(frame, 1)

            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_det.process(rgb)

            if res.detections:
                box = pick_largest_detection(res.detections, W=W, H=H)
                if box:
                    x, y, w, h = box
                    face = frame[y:y + h, x:x + w]
                    face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)

                    result = infer.infer(face)
                    emo = result.emotion if result.emotion else "Uncertain"
                    label = f"{emo} ({result.confidence:.2f}) [{result.backend}]"

                    # Colors (BGR) + per-state bbox color
                    TEXT_COLOR = (255, 255, 255)
                    BG_COLOR = (0, 0, 0)
                    BBOX_COLOR = (0, 0, 255) if result.is_uncertain else (0, 255, 0)

                    # BBOX
                    cv2.rectangle(frame, (x, y), (x + w, y + h), BBOX_COLOR, args.bbox_thickness)

                    # Text with background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = args.font_scale
                    text_thickness = args.text_thickness

                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                    tx = x
                    ty = max(text_h + 10, y - 10)

                    cv2.rectangle(
                        frame,
                        (tx, ty - text_h - baseline - 6),
                        (tx + text_w + 10, ty + 6),
                        BG_COLOR,
                        thickness=-1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (tx + 5, ty),
                        font,
                        font_scale,
                        TEXT_COLOR,
                        text_thickness,
                        cv2.LINE_AA
                    )

                    # Save JSON record
                    if do_save_json:
                        t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
                        if session_start_ms is None:
                            session_start_ms = t_ms
                        t_rel = max(0, t_ms - session_start_ms)

                        records.append({
                            "time_ms": t_ms,
                            "t_rel_ms": t_rel,
                            "timestamp": format_time_from_ms(t_rel),
                            "emotion": emo,
                            "confidence": float(result.confidence),
                            "backend": str(result.backend),
                            "is_uncertain": bool(result.is_uncertain),
                            "bbox": [int(x), int(y), int(w), int(h)],
                        })
            else:
                cv2.putText(
                    frame, "No face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
                )

            # Save annotated video (write BEFORE resize)
            if video_writer is not None:
                video_writer.write(frame)

            # Optional resize for display only
            if display_width and display_width > 0:
                scale = display_width / float(W)
                frame = cv2.resize(frame, (display_width, int(H * scale)))

            cv2.imshow("VideoEmotion Realtime (HSE only) - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()

    # Write JSON at end
    if do_save_json and json_path is not None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"session": session_ts, "records": records}, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON sauvegardé: {json_path}")

    # Video message
    if do_save_video and video_writer is not None and video_path is not None:
        print(f"[OK] Vidéo sauvegardée: {video_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
