# src/offline/visualize_results.py
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import cv2
import subprocess
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize emotion results on a video (overlay)."
    )

    p.add_argument("--video", required=True, help="Path to original input video")
    p.add_argument(
        "--results",
        required=True,
        help="Path to per-video detailed JSON (frame-level results)",
    )
    p.add_argument(
        "--out",
        required=True,
        help=(
            "Output path. If it's a directory, files are created inside it. "
            "If it's a .mp4 file, it will still be placed inside a per-video subfolder."
        ),
    )

    p.add_argument(
        "--bboxes-json",
        default=None,
        help="Optional JSON containing bbox per (frame_index, track_id). If provided, draws rectangles.",
    )

    p.add_argument(
        "--use-smoothed",
        action="store_true",
        help="Use smoothed_final_emotion when available",
    )
    p.add_argument(
        "--max-persons",
        type=int,
        default=8,
        help="Max persons to display as text per frame",
    )
    p.add_argument(
        "--show-confidence", action="store_true", help="Show confidence next to emotion"
    )
    p.add_argument(
        "--show-backend",
        action="store_true",
        help="Show backend used (hsemotion/deepface)",
    )
    p.add_argument("--font-scale", type=float, default=0.7, help="Text font scale")
    p.add_argument("--thickness", type=int, default=2, help="Text/rectangle thickness")

    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Force output FPS (default: input video FPS)",
    )
    p.add_argument(
        "--limit-frames", type=int, default=None, help="Debug: stop after N frames"
    )

    # --- alignment between video FPS and bbox/results FPS ---
    p.add_argument(
        "--bbox-fps",
        type=float,
        default=5.0,
        help="FPS at which bboxes JSON was generated (default: 5).",
    )
    p.add_argument(
        "--results-fps",
        type=float,
        default=None,
        help=(
            "FPS at which results JSON frame_index is defined. "
            "If omitted, defaults to bbox-fps (common when both come from extracted frames)."
        ),
    )
    p.add_argument(
        "--bbox-tolerance",
        type=int,
        default=1,
        help="Try +/- N frames in bbox/results index space when exact key is missing (default: 1).",
    )
    p.add_argument(
        "--bbox-offset",
        type=int,
        default=0,
        help="Optional offset applied after mapping video frame -> bbox/results frame index (default: 0).",
    )

    # --- H264 export ---
    p.add_argument(
        "--no-h264", action="store_true", help="Disable ffmpeg transcode to *_h264.mp4"
    )
    p.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="(Optional) ffmpeg command or full path. If not found, auto-detect with shutil.which().",
    )

    # --- copy inputs into output folder ---
    p.add_argument(
        "--copy-jsons",
        action="store_true",
        help="Copy results.json and bboxes.json (if provided) into the per-video output folder.",
    )

    return p.parse_args()


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_video_frame_to_index(
    fi_video: int, video_fps: float, data_fps: float, offset: int
) -> int:
    if video_fps <= 0:
        video_fps = 25.0
    if data_fps <= 0:
        data_fps = video_fps
    return int(round(fi_video * (data_fps / video_fps))) + int(offset)


def index_results_by_frame(
    results: Dict[str, Any], use_smoothed: bool
) -> Dict[int, List[Dict[str, Any]]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for _, entry in results.items():
        if not isinstance(entry, dict):
            continue
        fi = entry.get("frame_index")
        if fi is None:
            continue

        if use_smoothed and entry.get("smoothed_final_emotion") is not None:
            entry["_viz_emotion"] = entry.get("smoothed_final_emotion")
            entry["_viz_conf"] = entry.get("final_confidence", 0.0)
        else:
            entry["_viz_emotion"] = entry.get("final_emotion")
            entry["_viz_conf"] = entry.get("final_confidence", 0.0)

        by_frame[int(fi)].append(entry)

    for fi in list(by_frame.keys()):
        by_frame[fi].sort(
            key=lambda e: (e.get("track_id", 999999), e.get("face_id", 999999))
        )

    return by_frame


def load_bboxes(
    path: Optional[str],
) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}

    if isinstance(data, dict):
        for frame_key, items in data.items():
            fk = str(frame_key).replace("frame_", "")
            try:
                fi = int(fk)
            except ValueError:
                continue

            if isinstance(items, dict):
                for tid_str, bbox in items.items():
                    try:
                        tid = int(tid_str)
                    except ValueError:
                        continue
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x, y, w, h = map(int, bbox)
                        out[(fi, tid)] = (x, y, w, h)
                continue

            if not isinstance(items, list):
                continue

            for it in items:
                if not isinstance(it, dict):
                    continue
                tid = it.get("track_id")
                if tid is None:
                    continue
                if (
                    "bbox" in it
                    and isinstance(it["bbox"], list)
                    and len(it["bbox"]) == 4
                ):
                    x, y, w, h = map(int, it["bbox"])
                    out[(fi, int(tid))] = (x, y, w, h)
                else:
                    if all(k in it for k in ("x", "y", "w", "h")):
                        x, y, w, h = (
                            int(it["x"]),
                            int(it["y"]),
                            int(it["w"]),
                            int(it["h"]),
                        )
                        out[(fi, int(tid))] = (x, y, w, h)

    return out


def draw_text_lines(
    frame,
    lines: List[str],
    font_scale: float,
    thickness: int,
    origin: Tuple[int, int] = (15, 30),
    line_height: int = 28,
):
    x0, y0 = origin
    y = y0
    for s in lines:
        cv2.putText(
            frame,
            s,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def _resolve_ffmpeg(ffmpeg_path_arg: str) -> Optional[str]:
    """
    Returns an executable path for ffmpeg, or None if not found.
    Priority:
      1) explicit arg if it exists (full path) or is runnable
      2) shutil.which("ffmpeg")
    """
    # If user provided a full path, accept it if exists
    p = Path(ffmpeg_path_arg)
    if p.suffix.lower() == ".exe" and p.exists():
        return str(p)

    # If user provided "ffmpeg" but it's not in PATH, this will fail later;
    # we proactively try which().
    found = shutil.which("ffmpeg")
    if found:
        return found

    # As last resort, try the provided arg (maybe it's a command available)
    # but only if it's not empty
    if ffmpeg_path_arg:
        return ffmpeg_path_arg

    return None


def transcode_to_h264(ffmpeg_path_arg: str, src: Path, dst: Path) -> bool:
    ffmpeg_bin = _resolve_ffmpeg(ffmpeg_path_arg)
    if not ffmpeg_bin:
        print(
            "[WARN] ffmpeg introuvable: impossible de générer la version _h264.mp4 (la vidéo RAW est conservée)."
        )
        return False

    try:
        cmd = [
            ffmpeg_bin,
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
        subprocess.run(cmd, check=True)
        return dst.exists()
    except Exception as e:
        print(f"[WARN] Transcodage ffmpeg échoué: {e}")
        return False


def resolve_output_paths(video_path: Path, out_arg: str) -> Tuple[Path, Path]:
    out_p = Path(out_arg)

    if out_p.suffix.lower() == ".mp4":
        base_dir = out_p.parent
        raw_name = out_p.name
    else:
        base_dir = out_p
        raw_name = None

    video_stem = video_path.stem
    per_video_dir = base_dir / video_stem
    per_video_dir.mkdir(parents=True, exist_ok=True)

    if raw_name is None:
        raw_out = per_video_dir / f"{video_stem}_annotated_raw.mp4"
    else:
        raw_out = per_video_dir / raw_name

    return raw_out, per_video_dir


def try_get_bbox(
    bboxes: Dict[Tuple[int, int], Tuple[int, int, int, int]],
    fi_idx: int,
    tid: int,
    tol: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not bboxes:
        return None
    key = (fi_idx, tid)
    if key in bboxes:
        return bboxes[key]
    for d in range(1, max(0, tol) + 1):
        k1 = (fi_idx - d, tid)
        if k1 in bboxes:
            return bboxes[k1]
        k2 = (fi_idx + d, tid)
        if k2 in bboxes:
            return bboxes[k2]
    return None


def try_get_entries(
    by_frame: Dict[int, List[Dict[str, Any]]],
    fi_idx: int,
    tol: int,
) -> List[Dict[str, Any]]:
    if fi_idx in by_frame:
        return by_frame[fi_idx]
    for d in range(1, max(0, tol) + 1):
        if (fi_idx - d) in by_frame:
            return by_frame[fi_idx - d]
        if (fi_idx + d) in by_frame:
            return by_frame[fi_idx + d]
    return []


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results JSON not found: {args.results}")

    video_path = Path(args.video)
    raw_out_path, per_video_dir = resolve_output_paths(video_path, args.out)

    if args.copy_jsons:
        try:
            shutil.copy2(args.results, per_video_dir / Path(args.results).name)
        except Exception as e:
            print(f"[WARN] Could not copy results json: {e}")
        if args.bboxes_json:
            try:
                shutil.copy2(
                    args.bboxes_json, per_video_dir / Path(args.bboxes_json).name
                )
            except Exception as e:
                print(f"[WARN] Could not copy bboxes json: {e}")

    results = load_results(args.results)
    by_frame_results = index_results_by_frame(results, use_smoothed=args.use_smoothed)
    bboxes = load_bboxes(args.bboxes_json)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = args.fps if args.fps is not None else in_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    results_fps = args.results_fps if args.results_fps is not None else args.bbox_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(raw_out_path), fourcc, float(out_fps), (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {raw_out_path}")

    fi_video = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fi_results = map_video_frame_to_index(
            fi_video, in_fps, results_fps, args.bbox_offset
        )
        fi_bbox = map_video_frame_to_index(
            fi_video, in_fps, args.bbox_fps, args.bbox_offset
        )

        entries = try_get_entries(by_frame_results, fi_results, args.bbox_tolerance)
        lines: List[str] = []

        for e in entries[: max(0, args.max_persons)]:
            person = e.get("global_person_id") or f"track{e.get('track_id', '?')}"
            if isinstance(person, str) and "person_" in person:
                person = "person_" + person.split("person_")[-1]

            emo = e.get("_viz_emotion") or "unknown"
            conf = float(e.get("_viz_conf") or 0.0)
            backend = e.get("final_backend")

            s = f"{person}: {emo}"
            if args.show_confidence:
                s += f" ({conf:.2f})"
            if args.show_backend and backend:
                s += f" [{backend}]"
            lines.append(s)

        if bboxes and entries:
            for e in entries:
                tid = e.get("track_id")
                if tid is None:
                    continue

                bbox = try_get_bbox(bboxes, fi_bbox, int(tid), args.bbox_tolerance)
                if bbox is None:
                    continue

                x, y, w, h = bbox
                x2, y2 = x + w, y + h
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), args.thickness)

                emo = e.get("_viz_emotion") or "unknown"
                conf = float(e.get("_viz_conf") or 0.0)
                label = emo if not args.show_confidence else f"{emo} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    args.font_scale,
                    (255, 255, 255),
                    args.thickness,
                    cv2.LINE_AA,
                )

        if lines:
            draw_text_lines(
                frame, lines, font_scale=args.font_scale, thickness=args.thickness
            )

        out.write(frame)
        written += 1
        fi_video += 1

        if args.limit_frames is not None and written >= args.limit_frames:
            break

    cap.release()
    out.release()
    print(f"[OK] Wrote annotated RAW video: {raw_out_path} ({written} frames)")
    print(f"[INFO] Per-video output folder: {per_video_dir}")

    # ✅ Generate browser-friendly video, while keeping RAW
    if not args.no_h264:
        dst_out = raw_out_path.with_name(
            raw_out_path.stem.replace("_raw", "") + "_h264.mp4"
        )
        if raw_out_path.exists():
            if transcode_to_h264(args.ffmpeg_path, raw_out_path, dst_out):
                print(f"[OK] Wrote Chrome-friendly H264: {dst_out}")
            else:
                print(
                    "[WARN] La vidéo H264 n'a pas pu être générée (ffmpeg introuvable ou échec)."
                )


if __name__ == "__main__":
    main()
