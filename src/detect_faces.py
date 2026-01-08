# # detect_faces.py

# import os
# import cv2
# import mediapipe as mp
# import numpy as np
# from typing import List, Optional

# # Suppress TensorFlow logging
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # =============================================================================
# # CONFIG
# # =============================================================================

# # Tracking / matching
# OVERLAP_THRESHOLD = 0.30
# TRACK_MAX_MISSING = 60
# CENTER_DIST_MAX = 220          # base, on le rend dynamique
# BLUR_MIN = 40.0
# USE_APPEARANCE = True

# # --- Filtres détection (réduit les faux crops) ---
# MIN_DET_SCORE = 0.70          # score Mediapipe
# MIN_DET_SIZE = 80             # px : ignore bbox trop petite
# MAX_ASPECT_RATIO = 1.80       # bbox trop "plate" => souvent faux (oreille/main)

# # --- CONFIG ALIGNEMENT ---
# DESIRED_SIZE = (224, 224)
# EYE_DIST_RATIO = 0.33
# MAX_ROTATION_ANGLE = 30.0
# PROFILE_EYE_RATIO_THRESHOLD = 0.25

# # Filtre optionnel "yeux trop proches" (anti faux keypoints)
# ENABLE_EYE_RATIO_FILTER = True
# EYE_RATIO_MIN = 0.28          # ratio eye_dist / face_width

# # --- NOUVEAU : tolérance tracking (IMPORTANT pour David) ---
# DYN_CENTER_MULT = 3.0              # avant 1.8 (trop strict si bbox saute)
# MISSING_RELAX_PER_FRAME = 0.35     # +35% par frame manquante (jusqu'à cap)
# MISSING_RELAX_CAP = 2.5            # max x2.5 tolérance
# SINGLE_FACE_FALLBACK = True
# SINGLE_FACE_MAX_DIST_MULT = 6.0    # si 1 face/1 track, on force match si dist raisonnable

# # --- NOUVEAU : lissage bbox (réduit jitter Mediapipe) ---
# ENABLE_BBOX_SMOOTHING = True
# BBOX_SMOOTH_ALPHA = 0.70           # 0.7 = stable (0.5 plus réactif)

# # =============================================================================
# # UTILS
# # =============================================================================

# def blur_score(img: np.ndarray) -> float:
#     if img is None or img.size == 0:
#         return 0.0
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# def compute_face_hist(face_bgr: np.ndarray) -> Optional[np.ndarray]:
#     if face_bgr is None or face_bgr.size == 0:
#         return None
#     h = face_bgr.shape[0]
#     face = face_bgr[: int(h * 0.60), :]
#     try:
#         face = cv2.resize(face, (64, 64))
#         hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist
#     except Exception:
#         return None

# def hist_distance(h1, h2) -> float:
#     if h1 is None or h2 is None:
#         return 1.0
#     return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

# def center_of(box):
#     x, y, w, h = box
#     return (x + w / 2, y + h / 2)

# def center_distance(a, b) -> float:
#     ax, ay = a
#     bx, by = b
#     return float(np.hypot(ax - bx, ay - by))

# def iou(boxA, boxB) -> float:
#     # box: (x, y, w, h)
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
#     yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     interArea = interW * interH

#     areaA = max(0, boxA[2]) * max(0, boxA[3])
#     areaB = max(0, boxB[2]) * max(0, boxB[3])

#     denom = areaA + areaB - interArea
#     return (interArea / denom) if denom > 0 else 0.0

# def align_face(
#     image,
#     left_eye,
#     right_eye,
#     face_width,
#     desired_size=(224, 224),
#     left_eye_desired_ratio=0.33
# ):
#     """
#     Aligne le visage.
#     - Rejette si angle > MAX_ROTATION_ANGLE
#     - Rejette si profil détecté (ratio distance yeux / largeur visage trop faible)
#     - BORDER_REFLECT pour éviter bords noirs
#     """
#     dY = right_eye[1] - left_eye[1]
#     dX = right_eye[0] - left_eye[0]

#     eye_dist = np.sqrt((dX ** 2) + (dY ** 2))
#     if face_width <= 0:
#         return None

#     # Profil (yeux trop proches vs largeur bbox)
#     if (eye_dist / float(face_width)) < PROFILE_EYE_RATIO_THRESHOLD:
#         return None

#     angle = np.degrees(np.arctan2(dY, dX))
#     if abs(angle) > MAX_ROTATION_ANGLE:
#         return None

#     desired_right_eye_x = 1.0 - left_eye_desired_ratio
#     if eye_dist < 5:
#         return None

#     desired_dist = (desired_right_eye_x - left_eye_desired_ratio) * desired_size[0]
#     scale = desired_dist / eye_dist

#     eyes_center = (
#         (left_eye[0] + right_eye[0]) // 2,
#         (left_eye[1] + right_eye[1]) // 2
#     )

#     M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

#     tX = desired_size[0] * 0.5
#     tY = desired_size[1] * left_eye_desired_ratio
#     M[0, 2] += (tX - eyes_center[0])
#     M[1, 2] += (tY - eyes_center[1])

#     output = cv2.warpAffine(
#         image, M,
#         (desired_size[0], desired_size[1]),
#         flags=cv2.INTER_CUBIC,
#         borderMode=cv2.BORDER_REFLECT
#     )
#     return output

# # =============================================================================
# # TRACK CLASS
# # =============================================================================

# class Track:
#     def __init__(self, track_id, bbox, hist):
#         self.id = track_id
#         self.bbox = bbox
#         self.hist = hist
#         self.missing = 0

#         # bbox float for smoothing
#         self.bbox_f = np.array(bbox, dtype=np.float32)

# # =============================================================================
# # MAIN
# # =============================================================================

# def detect_faces_in_all_frames(extracted_frames_root, detected_faces_root):
#     mp_face = mp.solutions.face_detection

#     tracks: List[Track] = []
#     next_track_id = 0

#     # Reset tracking par séquence (ex: video_name/frames_timestamp)
#     current_sequence_key = None

#     with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
#         for dirpath, _, filenames in os.walk(extracted_frames_root):

#             rel_path = os.path.relpath(dirpath, extracted_frames_root)

#             # Séquence = dossier contenant les frames (chez toi, c'est rel_path)
#             sequence_key = rel_path

#             # Reset des tracks quand on change de séquence
#             if current_sequence_key is None:
#                 current_sequence_key = sequence_key
#             elif sequence_key != current_sequence_key:
#                 tracks = []
#                 next_track_id = 0
#                 current_sequence_key = sequence_key

#             out_base = os.path.join(detected_faces_root, rel_path)
#             os.makedirs(out_base, exist_ok=True)

#             for filename in sorted(filenames):
#                 if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
#                     continue

#                 img_path = os.path.join(dirpath, filename)
#                 image = cv2.imread(img_path)
#                 if image is None:
#                     continue

#                 h_img, w_img = image.shape[:2]
#                 rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 res = face_detection.process(rgb)

#                 detections = []
#                 if res.detections:
#                     for det in res.detections:
#                         # --- FILTRE 1: score mediapipe ---
#                         score = float(det.score[0]) if det.score else 0.0
#                         if score < MIN_DET_SCORE:
#                             continue

#                         rb = det.location_data.relative_bounding_box
#                         x = int(rb.xmin * w_img)
#                         y = int(rb.ymin * h_img)
#                         w = int(rb.width * w_img)
#                         h = int(rb.height * h_img)

#                         # clamp
#                         x, y = max(0, x), max(0, y)
#                         w = min(w, w_img - x)
#                         h = min(h, h_img - y)
#                         if w <= 0 or h <= 0:
#                             continue

#                         # --- FILTRE 2: taille bbox ---
#                         if w < MIN_DET_SIZE or h < MIN_DET_SIZE:
#                             continue

#                         # --- FILTRE 3: aspect ratio bbox ---
#                         aspect = max(w, h) / max(1, min(w, h))
#                         if aspect > MAX_ASPECT_RATIO:
#                             continue

#                         kp = det.location_data.relative_keypoints
#                         eye_left = (int(kp[0].x * w_img), int(kp[0].y * h_img))
#                         eye_right = (int(kp[1].x * w_img), int(kp[1].y * h_img))

#                         # --- FILTRE 4 (optionnel): ratio yeux / largeur bbox ---
#                         if ENABLE_EYE_RATIO_FILTER:
#                             dX = eye_right[0] - eye_left[0]
#                             dY = eye_right[1] - eye_left[1]
#                             eye_dist = float(np.sqrt(dX * dX + dY * dY))
#                             if (eye_dist / float(w)) < EYE_RATIO_MIN:
#                                 continue

#                         detections.append((x, y, w, h, eye_left, eye_right))

#                 used_tracks = set()

#                 for (x, y, w, h, eye_l, eye_r) in detections:
#                     # On tente l'alignement.
#                     # IMPORTANT: si align échoue, on SKIP pour éviter oreille/main => faux tracks
#                     final_crop = None
#                     try:
#                         final_crop = align_face(
#                             image, eye_l, eye_r, w,
#                             desired_size=DESIRED_SIZE,
#                             left_eye_desired_ratio=EYE_DIST_RATIO
#                         )
#                     except Exception:
#                         final_crop = None

#                     if final_crop is None:
#                         continue

#                     # Appearance (hist) seulement si assez net
#                     blur = blur_score(final_crop)
#                     hist = compute_face_hist(final_crop) if USE_APPEARANCE and blur >= BLUR_MIN else None

#                     new_box = (x, y, w, h)

#                     # Base threshold plus large + dépend de taille visage
#                     base_dyn = max(CENTER_DIST_MAX, int(DYN_CENTER_MULT * max(w, h)))

#                     best = None
#                     best_score = float("inf")

#                     for t in tracks:
#                         if t.id in used_tracks:
#                             continue

#                         dist = center_distance(center_of(t.bbox), center_of(new_box))
#                         ov = iou(t.bbox, new_box)

#                         # ✅ tolérance qui augmente si le track a raté des frames
#                         relax = 1.0 + MISSING_RELAX_PER_FRAME * min(t.missing, 6)
#                         relax = min(relax, MISSING_RELAX_CAP)
#                         allowed_dist = base_dyn * relax

#                         # plus tolérant sur overlap quand ça bouge beaucoup
#                         if dist > allowed_dist and ov < (OVERLAP_THRESHOLD * 0.6):
#                             continue

#                         # Score: distance - gros bonus overlap
#                         score = dist - 450.0 * ov

#                         # Apparence (si dispo)
#                         if hist is not None and t.hist is not None:
#                             score += 100.0 * hist_distance(hist, t.hist)

#                         if score < best_score:
#                             best_score = score
#                             best = t

#                     # ✅ fallback “single face”: si 1 track et 1 détection, on force match
#                     if best is None and SINGLE_FACE_FALLBACK and len(tracks) == 1 and len(detections) == 1:
#                         t0 = tracks[0]
#                         dist0 = center_distance(center_of(t0.bbox), center_of(new_box))
#                         if dist0 <= (SINGLE_FACE_MAX_DIST_MULT * max(w, h)):
#                             best = t0

#                     if best is not None:
#                         # Update bbox (avec smoothing optionnel)
#                         if ENABLE_BBOX_SMOOTHING:
#                             best.bbox_f = (
#                                 BBOX_SMOOTH_ALPHA * best.bbox_f
#                                 + (1.0 - BBOX_SMOOTH_ALPHA) * np.array(new_box, dtype=np.float32)
#                             )
#                             best.bbox = tuple(best.bbox_f.astype(int))
#                         else:
#                             best.bbox = new_box

#                         best.missing = 0
#                         if hist is not None:
#                             best.hist = hist

#                         track_id = best.id
#                     else:
#                         track_id = next_track_id
#                         tracks.append(Track(track_id, new_box, hist))
#                         next_track_id += 1

#                     person_dir = os.path.join(out_base, f"person_{track_id:04d}")
#                     os.makedirs(person_dir, exist_ok=True)

#                     out_name = filename.replace(".jpg", f"track{track_id:03d}.jpg")
#                     cv2.imwrite(os.path.join(person_dir, out_name), final_crop)

#                     used_tracks.add(track_id)

#                 # Update missing
#                 for t in tracks:
#                     if t.id not in used_tracks:
#                         t.missing += 1

#                 # Cleanup old tracks
#                 tracks = [t for t in tracks if t.missing <= TRACK_MAX_MISSING]


# if __name__ == "__main__":
#     detect_faces_in_all_frames(
#         "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\extracted_frames",
#         "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces",
#     )

# =========================
# detect_faces.py
# =========================
import os
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =============================================================================
# CONFIG
# =============================================================================

# Tracking / matching
OVERLAP_THRESHOLD = 0.30
TRACK_MAX_MISSING = 60
CENTER_DIST_MAX = 220
BLUR_MIN = 40.0
USE_APPEARANCE = True

# --- Filtres détection (réduit les faux crops) ---
MIN_DET_SCORE = 0.70
MIN_DET_SIZE = 80
MAX_ASPECT_RATIO = 1.80

# --- CONFIG ALIGNEMENT ---
DESIRED_SIZE = (224, 224)
EYE_DIST_RATIO = 0.33
MAX_ROTATION_ANGLE = 30.0
PROFILE_EYE_RATIO_THRESHOLD = 0.25

# Filtre optionnel "yeux trop proches" (anti faux keypoints)
ENABLE_EYE_RATIO_FILTER = True
EYE_RATIO_MIN = 0.28

# --- tolérance tracking ---
DYN_CENTER_MULT = 3.0
MISSING_RELAX_PER_FRAME = 0.35
MISSING_RELAX_CAP = 2.5
SINGLE_FACE_FALLBACK = True
SINGLE_FACE_MAX_DIST_MULT = 6.0

# --- lissage bbox ---
ENABLE_BBOX_SMOOTHING = True
BBOX_SMOOTH_ALPHA = 0.70

# =============================================================================
# UTILS
# =============================================================================

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
    left_eye_desired_ratio=0.33
):
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]

    eye_dist = np.sqrt((dX ** 2) + (dY ** 2))
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

    eyes_center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tX = desired_size[0] * 0.5
    tY = desired_size[1] * left_eye_desired_ratio
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    output = cv2.warpAffine(
        image, M,
        (desired_size[0], desired_size[1]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )
    return output

def sequence_already_processed(out_base: str) -> bool:
    """
    ✅ True si out_base contient au moins une image dans person_XXXX.
    """
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
# MAIN
# =============================================================================

def detect_faces_in_all_frames(extracted_frames_root, detected_faces_root):
    mp_face = mp.solutions.face_detection

    tracks: List[Track] = []
    next_track_id = 0

    current_sequence_key = None

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        for dirpath, _, filenames in os.walk(extracted_frames_root):

            rel_path = os.path.relpath(dirpath, extracted_frames_root)
            sequence_key = rel_path

            # Dossier de sortie pour cette séquence
            out_base = os.path.join(detected_faces_root, rel_path)

            # ✅ SKIP si déjà traité
            # (on skip tout le dossier de frames, et donc aucun nouveau track/timestamp)
            if sequence_already_processed(out_base):
                print(f"[SKIP] Faces déjà détectées pour la séquence: {rel_path}")
                continue

            # Reset des tracks quand on change de séquence
            if current_sequence_key is None:
                current_sequence_key = sequence_key
            elif sequence_key != current_sequence_key:
                tracks = []
                next_track_id = 0
                current_sequence_key = sequence_key

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
                        score = float(det.score[0]) if det.score else 0.0
                        if score < MIN_DET_SCORE:
                            continue

                        rb = det.location_data.relative_bounding_box
                        x = int(rb.xmin * w_img)
                        y = int(rb.ymin * h_img)
                        w = int(rb.width * w_img)
                        h = int(rb.height * h_img)

                        x, y = max(0, x), max(0, y)
                        w = min(w, w_img - x)
                        h = min(h, h_img - y)
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
                            if (eye_dist / float(w)) < EYE_RATIO_MIN:
                                continue

                        detections.append((x, y, w, h, eye_left, eye_right))

                used_tracks = set()

                for (x, y, w, h, eye_l, eye_r) in detections:
                    final_crop = None
                    try:
                        final_crop = align_face(
                            image, eye_l, eye_r, w,
                            desired_size=DESIRED_SIZE,
                            left_eye_desired_ratio=EYE_DIST_RATIO
                        )
                    except Exception:
                        final_crop = None

                    if final_crop is None:
                        continue

                    blur = blur_score(final_crop)
                    hist = compute_face_hist(final_crop) if USE_APPEARANCE and blur >= BLUR_MIN else None

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

                    if best is None and SINGLE_FACE_FALLBACK and len(tracks) == 1 and len(detections) == 1:
                        t0 = tracks[0]
                        dist0 = center_distance(center_of(t0.bbox), center_of(new_box))
                        if dist0 <= (SINGLE_FACE_MAX_DIST_MULT * max(w, h)):
                            best = t0

                    if best is not None:
                        if ENABLE_BBOX_SMOOTHING:
                            best.bbox_f = (
                                BBOX_SMOOTH_ALPHA * best.bbox_f
                                + (1.0 - BBOX_SMOOTH_ALPHA) * np.array(new_box, dtype=np.float32)
                            )
                            best.bbox = tuple(best.bbox_f.astype(int))
                        else:
                            best.bbox = new_box

                        best.missing = 0
                        if hist is not None:
                            best.hist = hist

                        track_id = best.id
                    else:
                        track_id = next_track_id
                        tracks.append(Track(track_id, new_box, hist))
                        next_track_id += 1

                    person_dir = os.path.join(out_base, f"person_{track_id:04d}")
                    os.makedirs(person_dir, exist_ok=True)

                    out_name = filename.replace(".jpg", f"track{track_id:03d}.jpg")
                    cv2.imwrite(os.path.join(person_dir, out_name), final_crop)

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
