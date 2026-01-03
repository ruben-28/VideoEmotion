import os
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =============================================================================
# CONFIG
# =============================================================================

OVERLAP_THRESHOLD = 0.3
TRACK_MAX_MISSING = 60          # très important pour 1 FPS
CENTER_DIST_MAX = 220           # px
BLUR_MIN = 40.0                 # seuil flou
HIST_MAX_DIST = 0.35            # Bhattacharyya (plus petit = plus strict)

USE_APPEARANCE = True

# =============================================================================
# UTILS
# =============================================================================

def blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_face_hist(face_bgr: np.ndarray) -> np.ndarray:
    """
    Histogramme HSV sur le HAUT du visage (anti-expression).
    """
    h = face_bgr.shape[0]
    face = face_bgr[: int(h * 0.60), :]  # haut du visage

    face = cv2.resize(face, (64, 64))
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_distance(h1, h2) -> float:
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


def center_of(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def center_distance(a, b) -> float:
    ax, ay = a
    bx, by = b
    return np.hypot(ax - bx, ay - by)

# =============================================================================
# TRACK CLASS
# =============================================================================

class Track:
    def __init__(self, track_id, bbox, hist):
        self.id = track_id
        self.bbox = bbox
        self.hist = hist
        self.missing = 0

# =============================================================================
# MAIN
# =============================================================================

def detect_faces_in_all_frames(extracted_frames_root, detected_faces_root):

    mp_face = mp.solutions.face_detection

    tracks: List[Track] = []
    next_track_id = 0

    with mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    ) as face_detection:

        for dirpath, _, filenames in os.walk(extracted_frames_root):
            rel_path = os.path.relpath(dirpath, extracted_frames_root)
            out_base = os.path.join(detected_faces_root, rel_path)
            os.makedirs(out_base, exist_ok=True)

            for filename in sorted(filenames):
                if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(dirpath, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                h_img, w_img = image.shape[:2]
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                res = face_detection.process(rgb)

                detections = []
                if res.detections:
                    for det in res.detections:
                        rb = det.location_data.relative_bounding_box
                        x = int(rb.xmin * w_img)
                        y = int(rb.ymin * h_img)
                        w = int(rb.width * w_img)
                        h = int(rb.height * h_img)
                        x, y = max(0, x), max(0, y)
                        w = min(w, w_img - x)
                        h = min(h, h_img - y)
                        detections.append((x, y, w, h))

                used_tracks = set()

                for (x, y, w, h) in detections:
                    crop = image[y:y+h, x:x+w]
                    blur = blur_score(crop)

                    hist = compute_face_hist(crop) if USE_APPEARANCE and blur >= BLUR_MIN else None
                    best = None
                    best_score = float("inf")

                    for t in tracks:
                        dist = center_distance(center_of(t.bbox), center_of((x, y, w, h)))
                        if dist > CENTER_DIST_MAX:
                            continue

                        score = dist
                        if hist is not None and t.hist is not None:
                            score += 100 * hist_distance(hist, t.hist)

                        if score < best_score:
                            best_score = score
                            best = t

                    if best:
                        best.bbox = (x, y, w, h)
                        best.missing = 0
                        if hist is not None:
                            best.hist = hist
                        track_id = best.id
                    else:
                        track_id = next_track_id
                        tracks.append(Track(track_id, (x, y, w, h), hist))
                        next_track_id += 1

                    person_dir = os.path.join(out_base, f"person_{track_id:04d}")
                    os.makedirs(person_dir, exist_ok=True)

                    out_name = filename.replace(".jpg", f"track{track_id:03d}.jpg")
                    cv2.imwrite(os.path.join(person_dir, out_name), crop)

                    used_tracks.add(track_id)

                for t in tracks:
                    if t.id not in used_tracks:
                        t.missing += 1

                tracks = [t for t in tracks if t.missing <= TRACK_MAX_MISSING]


if __name__ == "__main__":
    detect_faces_in_all_frames(
        "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\extracted_frames",
        "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces",
    )
