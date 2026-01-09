# src/visualize_results.py
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize emotion results on a video (overlay).")

    p.add_argument("--video", required=True, help="Path to original input video")
    p.add_argument("--results", required=True, help="Path to per-video detailed JSON (frame-level results)")
    p.add_argument("--out", required=True, help="Output annotated video path (.mp4)")

    # Optional: provide bbox metadata exported from detect_faces.py (recommended)
    p.add_argument(
        "--bboxes-json",
        default=None,
        help="Optional JSON containing bbox per (frame_index, track_id). If provided, draws rectangles.",
    )

    # Display options
    p.add_argument("--use-smoothed", action="store_true", help="Use smoothed_final_emotion when available")
    p.add_argument("--max-persons", type=int, default=8, help="Max persons to display as text per frame")
    p.add_argument("--show-confidence", action="store_true", help="Show confidence next to emotion")
    p.add_argument("--show-backend", action="store_true", help="Show backend used (hsemotion/deepface)")
    p.add_argument("--font-scale", type=float, default=0.7, help="Text font scale")
    p.add_argument("--thickness", type=int, default=2, help="Text/rectangle thickness")

    # Video options
    p.add_argument("--fps", type=float, default=None, help="Force output FPS (default: input video FPS)")
    p.add_argument("--limit-frames", type=int, default=None, help="Debug: stop after N frames")

    return p.parse_args()


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_results_by_frame(results: Dict[str, Any], use_smoothed: bool) -> Dict[int, List[Dict[str, Any]]]:
    """
    Returns: frame_index -> list of entries for that frame.
    """
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for k, entry in results.items():
        if not isinstance(entry, dict):
            continue
        fi = entry.get("frame_index")
        if fi is None:
            continue

        # pick emotion field
        if use_smoothed and entry.get("smoothed_final_emotion") is not None:
            entry["_viz_emotion"] = entry.get("smoothed_final_emotion")
            entry["_viz_conf"] = entry.get("final_confidence", 0.0)  # confidence stays final_confidence
        else:
            entry["_viz_emotion"] = entry.get("final_emotion")
            entry["_viz_conf"] = entry.get("final_confidence", 0.0)

        by_frame[int(fi)].append(entry)

    # sort each frame list by track_id then face_id for consistent display
    for fi in list(by_frame.keys()):
        by_frame[fi].sort(key=lambda e: (e.get("track_id", 999999), e.get("face_id", 999999)))

    return by_frame


def load_bboxes(path: Optional[str]) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Expected format (flexible), example:
    {
      "frame_0": [{"track_id": 0, "bbox": [x,y,w,h]}, ...],
      "1": [{"track_id": 0, "x":..., "y":..., "w":..., "h":...}]
    }
    Returns dict: (frame_index, track_id) -> (x,y,w,h)
    """
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}

    if isinstance(data, dict):
        for frame_key, items in data.items():
            # frame_key might be "frame_12" or "12"
            fk = str(frame_key).replace("frame_", "")
            try:
                fi = int(fk)
            except ValueError:
                continue

            if isinstance(items, dict):
                # sometimes direct mapping track_id -> bbox
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
                if "bbox" in it and isinstance(it["bbox"], list) and len(it["bbox"]) == 4:
                    x, y, w, h = map(int, it["bbox"])
                    out[(fi, int(tid))] = (x, y, w, h)
                else:
                    # x,y,w,h style
                    if all(k in it for k in ("x", "y", "w", "h")):
                        x, y, w, h = int(it["x"]), int(it["y"]), int(it["w"]), int(it["h"])
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
        cv2.putText(frame, s, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results JSON not found: {args.results}")

    results = load_results(args.results)
    by_frame = index_results_by_frame(results, use_smoothed=args.use_smoothed)
    bboxes = load_bboxes(args.bboxes_json)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = args.fps if args.fps is not None else in_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, out_fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {args.out}")

    fi = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        entries = by_frame.get(fi, [])
        lines: List[str] = []

        # TEXT overlay (always works)
        for e in entries[: max(0, args.max_persons)]:
            person = e.get("global_person_id") or f"track{e.get('track_id', '?')}"
            # shorten person label
            if isinstance(person, str) and "person_" in person:
                person = person.split("person_")[-1]
                person = "person_" + person

            emo = e.get("_viz_emotion") or "unknown"
            conf = float(e.get("_viz_conf") or 0.0)
            backend = e.get("final_backend")

            s = f"{person}: {emo}"
            if args.show_confidence:
                s += f" ({conf:.2f})"
            if args.show_backend and backend:
                s += f" [{backend}]"
            lines.append(s)

        # If bbox metadata exists, draw rectangles per (frame_index, track_id)
        if bboxes and entries:
            for e in entries:
                tid = e.get("track_id")
                if tid is None:
                    continue
                key = (fi, int(tid))
                if key not in bboxes:
                    continue
                x, y, w, h = bboxes[key]
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
            draw_text_lines(frame, lines, font_scale=args.font_scale, thickness=args.thickness)

        out.write(frame)
        written += 1
        fi += 1

        if args.limit_frames is not None and written >= args.limit_frames:
            break

    cap.release()
    out.release()
    print(f"[OK] Wrote annotated video: {args.out} ({written} frames)")

if __name__ == "__main__":
    main()
