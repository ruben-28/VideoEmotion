import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import re
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =============================================================================
# CONFIG (valeurs par défaut, override possible via config.yaml)
# =============================================================================

OVERLAP_THRESHOLD = 0.30
TRACK_MAX_MISSING = 60
CENTER_DIST_MAX = 220
BLUR_MIN = 40.0
USE_APPEARANCE = True

MIN_DET_SCORE = 0.70
MIN_DET_SIZE = 80
MAX_ASPECT_RATIO = 1.80

DESIRED_SIZE = (224, 224)
EYE_DIST_RATIO = 0.33
MAX_ROTATION_ANGLE = 30.0
PROFILE_EYE_RATIO_THRESHOLD = 0.25

ENABLE_EYE_RATIO_FILTER = True
EYE_RATIO_MIN = 0.28

DYN_CENTER_MULT = 3.0
MISSING_RELAX_PER_FRAME = 0.35
MISSING_RELAX_CAP = 2.5
SINGLE_FACE_FALLBACK = True
SINGLE_FACE_MAX_DIST_MULT = 6.0

ENABLE_BBOX_SMOOTHING = True
BBOX_SMOOTH_ALPHA = 0.70

# =============================================================================
# CONFIG HELPERS
# =============================================================================


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


def apply_config_overrides(cfg: Dict[str, Any]) -> None:
    """
    Override les constantes globales depuis config.yaml si présent.
    """
    global OVERLAP_THRESHOLD, TRACK_MAX_MISSING, CENTER_DIST_MAX
    global MIN_DET_SCORE, MIN_DET_SIZE, MAX_ASPECT_RATIO
    global ENABLE_BBOX_SMOOTHING, BBOX_SMOOTH_ALPHA
    global ENABLE_EYE_RATIO_FILTER, EYE_RATIO_MIN

    # tracking
    OVERLAP_THRESHOLD = float(
        cfg_get(
            cfg,
            "face_detection",
            "tracking",
            "iou_threshold",
            default=OVERLAP_THRESHOLD,
        )
    )
    TRACK_MAX_MISSING = int(
        cfg_get(
            cfg,
            "face_detection",
            "tracking",
            "max_missing_frames",
            default=TRACK_MAX_MISSING,
        )
    )
    CENTER_DIST_MAX = int(
        cfg_get(
            cfg,
            "face_detection",
            "tracking",
            "center_distance_max",
            default=CENTER_DIST_MAX,
        )
    )

    # filtres détection
    MIN_DET_SCORE = float(
        cfg_get(
            cfg, "face_detection", "filters", "min_det_score", default=MIN_DET_SCORE
        )
    )
    MIN_DET_SIZE = int(
        cfg_get(cfg, "face_detection", "filters", "min_det_size", default=MIN_DET_SIZE)
    )
    MAX_ASPECT_RATIO = float(
        cfg_get(
            cfg,
            "face_detection",
            "filters",
            "max_aspect_ratio",
            default=MAX_ASPECT_RATIO,
        )
    )

    # bbox smoothing
    ENABLE_BBOX_SMOOTHING = bool(
        cfg_get(
            cfg,
            "face_detection",
            "bbox_smoothing",
            "enabled",
            default=ENABLE_BBOX_SMOOTHING,
        )
    )
    BBOX_SMOOTH_ALPHA = float(
        cfg_get(
            cfg, "face_detection", "bbox_smoothing", "alpha", default=BBOX_SMOOTH_ALPHA
        )
    )

    # eye ratio filter
    ENABLE_EYE_RATIO_FILTER = bool(
        cfg_get(
            cfg,
            "face_detection",
            "filters",
            "enable_eye_ratio_filter",
            default=ENABLE_EYE_RATIO_FILTER,
        )
    )
    EYE_RATIO_MIN = float(
        cfg_get(
            cfg, "face_detection", "filters", "eye_ratio_min", default=EYE_RATIO_MIN
        )
    )


# =============================================================================
# UTILS
# =============================================================================

_FRAME_RE = re.compile(r"frame_(\d+)")


def parse_frame_index(filename: str) -> Optional[int]:
    """
    Extrait frame_index depuis un filename comme:
    frame_00001_t00000239.jpg
    frame_00001_t00000239track000.jpg
    """
    m = _FRAME_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def clip_box(
    x: int, y: int, w: int, h: int, W: int, H: int
) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return x, y, w, h


def blur_score(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_face_hist(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    if face_bgr is None or face_bgr.size == 0:
        return None
    h = face_bgr.shape[0]
    face = face_bgr[: int(h * 0.60), :]
    try:
        face = cv2.resize(face, (64, 64))
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
    except Exception:
        return None


def hist_distance(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 1.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def center_of(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def center_distance(a, b) -> float:
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))


def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = max(0, boxA[2]) * max(0, boxA[3])
    areaB = max(0, boxB[2]) * max(0, boxB[3])

    denom = areaA + areaB - interArea
    return (interArea / denom) if denom > 0 else 0.0


def align_face(
    image,
    left_eye,
    right_eye,
    face_width,
    desired_size=(224, 224),
    left_eye_desired_ratio=0.33,
):
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]

    eye_dist = np.sqrt((dX**2) + (dY**2))
    if face_width <= 0:
        return None

    if (eye_dist / float(face_width)) < PROFILE_EYE_RATIO_THRESHOLD:
        return None

    angle = np.degrees(np.arctan2(dY, dX))
    if abs(angle) > MAX_ROTATION_ANGLE:
        return None

    desired_right_eye_x = 1.0 - left_eye_desired_ratio
    if eye_dist < 5:
        return None

    desired_dist = (desired_right_eye_x - left_eye_desired_ratio) * desired_size[0]
    scale = desired_dist / eye_dist

    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tX = desired_size[0] * 0.5
    tY = desired_size[1] * left_eye_desired_ratio
    M[0, 2] += tX - eyes_center[0]
    M[1, 2] += tY - eyes_center[1]

    output = cv2.warpAffine(
        image,
        M,
        (desired_size[0], desired_size[1]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return output


def sequence_already_processed(out_base: str) -> bool:
    if not os.path.exists(out_base):
        return False
    for name in os.listdir(out_base):
        if name.lower().startswith("person_"):
            pdir = os.path.join(out_base, name)
            if os.path.isdir(pdir) and any(
                f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(pdir)
            ):
                return True
    return False


# =============================================================================
# TRACK CLASS
# =============================================================================


class Track:
    def __init__(self, track_id, bbox, hist):
        self.id = track_id
        self.bbox = bbox
        self.hist = hist
        self.missing = 0
        self.bbox_f = np.array(bbox, dtype=np.float32)


# =============================================================================
# BBOX EXPORT
# =============================================================================


def write_bboxes_json(
    out_base: str, bbox_by_frame: Dict[int, List[Dict[str, Any]]], filename: str
) -> None:
    """
    Ecrit un JSON du type:
    {
      "0": [{"track_id": 0, "bbox": [x,y,w,h]}],
      "1": [{"track_id": 0, "bbox": [x,y,w,h]}, {"track_id": 1, "bbox": [...] }]
    }
    """
    if not bbox_by_frame:
        return
    out_path = os.path.join(out_base, filename)
    serializable = {
        str(k): v for k, v in sorted(bbox_by_frame.items(), key=lambda kv: kv[0])
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


# =============================================================================
# MAIN LOGIC
# =============================================================================


def detect_faces_in_all_frames(
    extracted_frames_root: str,
    detected_faces_root: str,
    export_bboxes: bool = False,
    bboxes_name: str = "bboxes.json",
) -> None:
    mp_face = mp.solutions.face_detection

    tracks: List[Track] = []
    next_track_id = 0
    current_sequence_key = None

    with mp_face.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        for dirpath, _, filenames in os.walk(extracted_frames_root):
            rel_path = os.path.relpath(dirpath, extracted_frames_root)
            sequence_key = rel_path

            out_base = os.path.join(detected_faces_root, rel_path)
            bboxes_path = os.path.join(out_base, bboxes_name)

            already = sequence_already_processed(out_base)

            # ✅ SKIP si déjà traité
            # Mais si export_bboxes demandé et bboxes.json absent => on ne skip pas.
            if already and (not export_bboxes or os.path.exists(bboxes_path)):
                print(f"[SKIP] Faces déjà détectées pour la séquence: {rel_path}")
                continue

            # Reset tracks quand on change de séquence
            if current_sequence_key is None:
                current_sequence_key = sequence_key
            elif sequence_key != current_sequence_key:
                tracks = []
                next_track_id = 0
                current_sequence_key = sequence_key

            os.makedirs(out_base, exist_ok=True)

            # ---- BBOX records pour cette séquence ----
            bbox_by_frame: Dict[int, List[Dict[str, Any]]] = {}

            for filename in sorted(filenames):
                if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(dirpath, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                frame_index = parse_frame_index(filename)
                # si pas trouvé, on ignore juste l'export bbox pour cette frame
                # mais on continue la détection/crops normalement
                h_img, w_img = image.shape[:2]

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                res = face_detection.process(rgb)

                detections = []
                if res.detections:
                    for det in res.detections:
                        score = float(det.score[0]) if det.score else 0.0
                        if score < MIN_DET_SCORE:
                            continue

                        rb = det.location_data.relative_bounding_box
                        x = int(rb.xmin * w_img)
                        y = int(rb.ymin * h_img)
                        w = int(rb.width * w_img)
                        h = int(rb.height * h_img)

                        x, y, w, h = clip_box(x, y, w, h, w_img, h_img)
                        if w <= 0 or h <= 0:
                            continue

                        if w < MIN_DET_SIZE or h < MIN_DET_SIZE:
                            continue

                        aspect = max(w, h) / max(1, min(w, h))
                        if aspect > MAX_ASPECT_RATIO:
                            continue

                        kp = det.location_data.relative_keypoints
                        eye_left = (int(kp[0].x * w_img), int(kp[0].y * h_img))
                        eye_right = (int(kp[1].x * w_img), int(kp[1].y * h_img))

                        if ENABLE_EYE_RATIO_FILTER:
                            dX = eye_right[0] - eye_left[0]
                            dY = eye_right[1] - eye_left[1]
                            eye_dist = float(np.sqrt(dX * dX + dY * dY))
                            if (eye_dist / float(max(1, w))) < EYE_RATIO_MIN:
                                continue

                        detections.append((x, y, w, h, eye_left, eye_right))

                used_tracks = set()

                for x, y, w, h, eye_l, eye_r in detections:
                    final_crop = None
                    try:
                        final_crop = align_face(
                            image,
                            eye_l,
                            eye_r,
                            w,
                            desired_size=DESIRED_SIZE,
                            left_eye_desired_ratio=EYE_DIST_RATIO,
                        )
                    except Exception:
                        final_crop = None

                    if final_crop is None:
                        continue

                    blur = blur_score(final_crop)
                    hist = (
                        compute_face_hist(final_crop)
                        if USE_APPEARANCE and blur >= BLUR_MIN
                        else None
                    )

                    new_box = (x, y, w, h)
                    base_dyn = max(CENTER_DIST_MAX, int(DYN_CENTER_MULT * max(w, h)))

                    best = None
                    best_score = float("inf")

                    for t in tracks:
                        if t.id in used_tracks:
                            continue

                        dist = center_distance(center_of(t.bbox), center_of(new_box))
                        ov = iou(t.bbox, new_box)

                        relax = 1.0 + MISSING_RELAX_PER_FRAME * min(t.missing, 6)
                        relax = min(relax, MISSING_RELAX_CAP)
                        allowed_dist = base_dyn * relax

                        if dist > allowed_dist and ov < (OVERLAP_THRESHOLD * 0.6):
                            continue

                        score = dist - 450.0 * ov
                        if hist is not None and t.hist is not None:
                            score += 100.0 * hist_distance(hist, t.hist)

                        if score < best_score:
                            best_score = score
                            best = t

                    if (
                        best is None
                        and SINGLE_FACE_FALLBACK
                        and len(tracks) == 1
                        and len(detections) == 1
                    ):
                        t0 = tracks[0]
                        dist0 = center_distance(center_of(t0.bbox), center_of(new_box))
                        if dist0 <= (SINGLE_FACE_MAX_DIST_MULT * max(w, h)):
                            best = t0

                    if best is not None:
                        if ENABLE_BBOX_SMOOTHING:
                            best.bbox_f = BBOX_SMOOTH_ALPHA * best.bbox_f + (
                                1.0 - BBOX_SMOOTH_ALPHA
                            ) * np.array(new_box, dtype=np.float32)
                            best.bbox = tuple(best.bbox_f.astype(int))
                        else:
                            best.bbox = new_box

                        best.missing = 0
                        if hist is not None:
                            best.hist = hist

                        track_id = best.id
                        bbox_for_frame = best.bbox  # bbox possiblement lissée
                    else:
                        track_id = next_track_id
                        tracks.append(Track(track_id, new_box, hist))
                        next_track_id += 1
                        bbox_for_frame = new_box

                    # ---- SAVE CROP ----
                    person_dir = os.path.join(out_base, f"person_{track_id:04d}")
                    os.makedirs(person_dir, exist_ok=True)

                    out_name = filename.replace(".jpg", f"track{track_id:03d}.jpg")
                    cv2.imwrite(os.path.join(person_dir, out_name), final_crop)

                    # ---- SAVE BBOX METADATA ----
                    if export_bboxes and frame_index is not None:
                        bx, by, bw, bh = bbox_for_frame
                        bx, by, bw, bh = clip_box(
                            int(bx), int(by), int(bw), int(bh), w_img, h_img
                        )
                        bbox_by_frame.setdefault(int(frame_index), []).append(
                            {"track_id": int(track_id), "bbox": [bx, by, bw, bh]}
                        )

                    used_tracks.add(track_id)

                # Update missing
                for t in tracks:
                    if t.id not in used_tracks:
                        t.missing += 1

                tracks = [t for t in tracks if t.missing <= TRACK_MAX_MISSING]

            # ---- Write bbox JSON at end of this sequence ----
            if export_bboxes:
                write_bboxes_json(out_base, bbox_by_frame, bboxes_name)
                print(f"[OK] BBOX export: {os.path.join(out_base, bboxes_name)}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces from extracted frames (VideoEmotion)"
    )
    parser.add_argument(
        "--input-frames", default=None, help="Override input frames root."
    )
    parser.add_argument(
        "--output-faces", default=None, help="Override output faces root."
    )
    parser.add_argument(
        "--project-root", default=None, help="Racine du projet (défaut: auto)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Chemin vers config.yaml (défaut: <project-root>/config.yaml).",
    )

    # NEW: bbox export
    parser.add_argument(
        "--export-bboxes",
        action="store_true",
        help="Export bboxes.json per sequence folder.",
    )
    parser.add_argument(
        "--bboxes-name",
        default="bboxes.json",
        help="Bboxes json filename inside each sequence output folder.",
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

    apply_config_overrides(cfg)

    cfg_in = cfg_get(cfg, "paths", "extracted_frames", default="data/extracted_frames")
    cfg_out = cfg_get(cfg, "paths", "detected_faces", default="data/detected_faces")

    extracted_frames_root = (
        resolve_from_project(project_root, args.input_frames)
        if args.input_frames
        else resolve_from_project(project_root, str(cfg_in))
    )
    detected_faces_root = (
        resolve_from_project(project_root, args.output_faces)
        if args.output_faces
        else resolve_from_project(project_root, str(cfg_out))
    )

    if not extracted_frames_root.exists():
        print(f"[ERREUR] Dossier frames introuvable: {extracted_frames_root}")
        return

    detected_faces_root.mkdir(parents=True, exist_ok=True)

    detect_faces_in_all_frames(
        str(extracted_frames_root),
        str(detected_faces_root),
        export_bboxes=bool(args.export_bboxes),
        bboxes_name=str(args.bboxes_name),
    )


if __name__ == "__main__":
    main()

# commande to run:
# mp_env\Scripts\python.exe src/detect_faces.py --export-bboxes
