import os
import re
import cv2
import csv
import json
import traceback
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

# =============================================================================
# GPU SETUP (IMPORTANT)
# =============================================================================
# 1) HSEmotion (PyTorch) : device="cuda" si dispo
# 2) DeepFace (TensorFlow) : on active "memory growth" pour éviter OOM
#
# NOTE: ceci ne change pas tes choix: TTA reste ON, backend reste retinaface, enforce_detection reste True.

def _setup_tensorflow_gpu():
    """
    Configure TensorFlow pour utiliser le GPU proprement (memory growth).
    Si TensorFlow n'est pas installé / pas GPU, ça ne casse pas le script.
    """
    try:
        import tensorflow as tf  # DeepFace dépend souvent de TF

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[GPU][TF] Aucun GPU détecté par TensorFlow (DeepFace tournera probablement CPU).")
            return

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        names = [g.name for g in gpus]
        print(f"[GPU][TF] GPU(s) TensorFlow détecté(s): {names} (memory growth activé)")
    except Exception:
        print("[GPU][TF] TensorFlow non configuré (ou non présent). DeepFace peut tourner CPU.")
        # Pas d'exception bloquante


def _select_hse_device(preferred: str = "cuda") -> str:
    """
    Retourne 'cuda' si PyTorch voit un GPU, sinon 'cpu'.
    """
    try:
        import torch
        if preferred == "cuda" and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU][Torch] GPU détecté: {name}")
            return "cuda"
        print("[GPU][Torch] GPU non dispo -> CPU")
        return "cpu"
    except Exception:
        print("[GPU][Torch] PyTorch non dispo -> CPU")
        return "cpu"


# Appeler la config TF GPU le plus tôt possible
_setup_tensorflow_gpu()

from deepface import DeepFace  # après setup TF


# =============================================================================
# CONFIG
# =============================================================================

# --- 1) Crop quality / filtering ---
MIN_FACE_SIZE = 60  # px : ignore les faces trop petites (souvent bruitées)

# --- 2) Face quality metrics thresholds (utilisés pour flag "bad_quality") ---
BLUR_MIN_VAR_LAPLACIAN = 40.0   # < ~40 => souvent flou (à ajuster)
BRIGHTNESS_MIN = 35.0           # trop sombre
BRIGHTNESS_MAX = 220.0          # trop surexposé

# --- 3) DeepFace detector backend (comme tu veux) ---
DEEPFACE_DETECTOR_BACKEND = "retinaface"   # tu veux garder retinaface
DEEPFACE_ENFORCE_DETECTION = True          # tu veux garder True

# --- 4) TTA (Test-Time Augmentation) ---
ENABLE_TTA = True
TTA_ROT_DEG = 5
TTA_BRIGHT_FACTOR = 0.10  # +/- 10%
TTA_MAX_VARIANTS = 5      # garde petit (perf)

# --- 5) Temporal smoothing centré (n-1, n, n+1) ---
ENABLE_SMOOTHING = True
CENTERED_SMOOTH_WINDOW = 3  # IMPORTANT: doit être impair (3 => n-1,n,n+1)

# On calcule TOUJOURS HSEmotion. DeepFace: soit toujours, soit fallback.
RUN_DEEPFACE_ALWAYS = False  # si False: DeepFace seulement si HSEmotion faible.

# Seuils de fallback (si tu veux garder le "final_emotion" comme avant)
HSEMOTION_CONFIDENCE_THRESHOLD = 0.65
DEEPFACE_CONFIDENCE_THRESHOLD = 0.70

# "Uncertain" (D)
ENABLE_UNCERTAIN_CLASS = True
UNCERTAIN_MIN_CONF = 0.55  # si les deux < ceci => incertain


# =============================================================================
# UTIL PARSING
# =============================================================================

def parse_frame_index(filename: str) -> int:
    """
    Supporte:
      frame_00012face000.jpg / frame_00012_face_000.jpg
      frame_00012track000.jpg / frame_00012_track_000.jpg
    """
    m = re.search(r"frame_(\d+)", filename.lower())
    return int(m.group(1)) if m else -1


def parse_face_id(filename: str) -> int:
    """
    Extrait un id depuis ...faceXYZ.jpg ou ...trackXYZ.jpg
    Ex:
      frame_00012face003.jpg  -> 3
      frame_00012track003.jpg -> 3
    Retourne -1 si introuvable.
    """
    try:
        lower = filename.lower()

        for key in ("track", "face"):
            idx = lower.rfind(key)
            if idx != -1:
                tail = lower[idx + len(key):]
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                return int(digits) if digits else -1

        return -1
    except Exception:
        return -1


def parse_track_id(filename: str) -> int:
    """
    Extrait l'ID du tracking depuis ...trackXYZ.jpg (nouveau format)
    Ex: frame_00012track007.jpg -> 7
    Retourne -1 si introuvable.
    """
    try:
        lower = filename.lower()
        idx = lower.rfind("track")
        if idx == -1:
            return -1
        tail = lower[idx + 5:]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else -1
    except Exception:
        return -1


def identity_id(face_id: int, track_id: int) -> int:
    """
    ID "stable" utilisé pour smoothing/tracking.
    - Si track_id existe => on l'utilise
    - Sinon fallback sur face_id
    """
    return track_id if track_id != -1 else face_id


# =============================================================================
# MASTER JSON
# =============================================================================

def load_master_json(master_json_path: str) -> dict:
    if os.path.exists(master_json_path):
        with open(master_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_master_json(master: dict, master_json_path: str):
    os.makedirs(os.path.dirname(master_json_path), exist_ok=True)
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=4, ensure_ascii=False)


# =============================================================================
# QUALITY METRICS
# =============================================================================

def face_quality_metrics(face_bgr: np.ndarray) -> Dict[str, float]:
    """
    Qualité de crop : blur/brightness/contrast/area
    """
    if face_bgr is None or face_bgr.size == 0:
        return {"blur": 0.0, "brightness": 0.0, "contrast": 0.0, "area": 0.0}

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    area = float(face_bgr.shape[0] * face_bgr.shape[1])
    return {
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "area": area,
    }


def is_bad_quality(q: Dict[str, float]) -> bool:
    """
    Flag utile pour debug/export : pas forcément pour “rejeter”.
    """
    if q["area"] <= 0:
        return True
    if q["blur"] < BLUR_MIN_VAR_LAPLACIAN:
        return True
    if q["brightness"] < BRIGHTNESS_MIN or q["brightness"] > BRIGHTNESS_MAX:
        return True
    return False


# =============================================================================
# TTA
# =============================================================================

def tta_variants(face_bgr: np.ndarray) -> List[np.ndarray]:
    """
    TTA léger : original, flip, +/- luminosité, micro-rotation
    """
    variants = [face_bgr]
    if face_bgr is None or face_bgr.size == 0:
        return variants

    # flip
    variants.append(cv2.flip(face_bgr, 1))

    # brightness +/- 10%
    img = face_bgr.astype(np.float32)
    brighter = np.clip(img * (1.0 + TTA_BRIGHT_FACTOR), 0, 255).astype(np.uint8)
    darker = np.clip(img * (1.0 - TTA_BRIGHT_FACTOR), 0, 255).astype(np.uint8)
    variants.append(brighter)
    variants.append(darker)

    # rotation
    h, w = face_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), TTA_ROT_DEG, 1.0)
    rot = cv2.warpAffine(
        face_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    variants.append(rot)

    return variants[:TTA_MAX_VARIANTS]


# =============================================================================
# CENTERED SMOOTHING (n-1, n, n+1)
# =============================================================================

def centered_mode(values: List[Optional[str]]) -> Optional[str]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def apply_centered_smoothing(
    entries: List[dict],
    window: int,
    key_emotion: str,
    out_key: str,
):
    """
    Smoothing centré : fenêtre impaire (3 => n-1,n,n+1).
    entries doit être trié par frame_index.
    """
    if window % 2 != 1 or window < 3:
        raise ValueError("window doit être impair et >= 3 (ex: 3,5,7...)")

    half = window // 2
    n = len(entries)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals = [entries[j].get(key_emotion) for j in range(lo, hi)]
        entries[i][out_key] = centered_mode(vals)


def apply_centered_smoothing_per_dir(per_dir_rows: Dict[str, List[dict]], window: int):
    """
    Applique smoothing centré par (rel_dir, identity_id),
    sur hse_emotion + deepface_emotion puis calcule smoothed_final_emotion.
    """
    for rel_dir, rows in per_dir_rows.items():
        # group par identity_id
        groups: Dict[int, List[dict]] = {}
        for r in rows:
            iid = int(r.get("identity_id", -1))
            groups.setdefault(iid, []).append(r)

        for iid, lst in groups.items():
            lst.sort(key=lambda x: int(x.get("frame_index", -1)))

            apply_centered_smoothing(
                lst,
                window=window,
                key_emotion="hse_emotion",
                out_key="smoothed_hse_emotion",
            )
            apply_centered_smoothing(
                lst,
                window=window,
                key_emotion="deepface_emotion",
                out_key="smoothed_deepface_emotion",
            )

            for r in lst:
                sh = r.get("smoothed_hse_emotion")
                sd = r.get("smoothed_deepface_emotion")
                if sh is not None and sd is not None:
                    r["smoothed_final_emotion"] = sh if sh == sd else sh
                else:
                    r["smoothed_final_emotion"] = sh or sd


# =============================================================================
# BACKEND 1: DEEPFACE
# =============================================================================

class DeepFaceEmotionDetector:
    """
    DeepFace avec backend robuste (retinaface/mediapipe)
    + support TTA (optionnel)
    """

    def __init__(
        self,
        detector_backend: str = DEEPFACE_DETECTOR_BACKEND,
        enforce_detection: bool = DEEPFACE_ENFORCE_DETECTION,
    ):
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        print(
            f"[DeepFaceEmotionDetector] Ready (backend={detector_backend}, enforce={enforce_detection})"
        )

        # Warm-up optionnel (réduit souvent le coût du 1er appel)
        # On évite de chauffer avec retinaface sur un vrai crop ici, mais tu peux décommenter si tu veux.
        # self._warmup_done = False

    def analyze_once(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pred = DeepFace.analyze(
            img_rgb,
            actions=["emotion"],
            enforce_detection=self.enforce_detection,
            detector_backend=self.detector_backend,
            silent=True,
        )

        res0 = pred[0] if isinstance(pred, list) and pred else pred
        if not isinstance(res0, dict):
            return None, 0.0

        dominant = res0.get("dominant_emotion", None)
        scores = res0.get("emotion", {}) or {}
        raw = float(scores.get(dominant, 0.0)) if dominant else 0.0

        conf = raw / 100.0 if raw > 1.5 else raw
        return dominant, conf

    def analyze(self, img_bgr: np.ndarray, use_tta: bool = False) -> Tuple[Optional[str], float]:
        if not use_tta:
            try:
                return self.analyze_once(img_bgr)
            except Exception:
                return None, 0.0

        emotions = []
        confs = []
        for v in tta_variants(img_bgr):
            try:
                e, c = self.analyze_once(v)
                if e is not None:
                    emotions.append(e)
                    confs.append(float(c))
            except Exception:
                continue

        if not emotions:
            return None, 0.0

        top = Counter(emotions).most_common(1)[0][0]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return top, mean_conf


# =============================================================================
# BACKEND 2: HSEMOTION
# =============================================================================

class HSEmotionDetector:
    """
    Wrapper HSEmotion.
    Retourne (emotion, confidence) proba max (0..1).
    """

    def __init__(self, device: str = "cpu"):
        self._printed_error = False
        print(f"[HSEmotionDetector] Chargement du modèle HSEmotion (device={device})...")
        from hsemotion.facial_emotions import HSEmotionRecognizer

        self.model = HSEmotionRecognizer(
            model_name="enet_b0_8_best_vgaf", device=device
        )
        print("[HSEmotionDetector] Modèle chargé ✅")

    def analyze_once(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        emotion, scores = self.model.predict_emotions(img_rgb, logits=False)
        conf = float(np.max(scores)) if scores is not None else 0.0
        return emotion, conf

    def analyze(self, img_bgr: np.ndarray, use_tta: bool = False) -> Tuple[Optional[str], float]:
        if not use_tta:
            try:
                return self.analyze_once(img_bgr)
            except Exception:
                if not self._printed_error:
                    self._printed_error = True
                    print("[HSEmotionDetector] ERREUR (une seule fois) :")
                    traceback.print_exc()
                return None, 0.0

        emotions = []
        confs = []
        for v in tta_variants(img_bgr):
            try:
                e, c = self.analyze_once(v)
                if e is not None:
                    emotions.append(e)
                    confs.append(float(c))
            except Exception:
                continue

        if not emotions:
            return None, 0.0

        top = Counter(emotions).most_common(1)[0][0]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return top, mean_conf


# =============================================================================
# DECISION LOGIC (final + uncertain)
# =============================================================================

def decide_final(
    hse_emotion: Optional[str],
    hse_conf: float,
    df_emotion: Optional[str],
    df_conf: float,
    quality_bad: bool,
) -> Tuple[Optional[str], float, str, bool]:
    """
    Retourne:
      final_emotion, final_confidence, final_backend, is_uncertain
    """

    if ENABLE_UNCERTAIN_CLASS:
        low_both = (hse_conf < UNCERTAIN_MIN_CONF) and (df_conf < UNCERTAIN_MIN_CONF)
        disagree = (
            hse_emotion is not None
            and df_emotion is not None
            and hse_emotion != df_emotion
        )

        if low_both or (quality_bad and disagree):
            return None, 0.0, "uncertain", True

    if hse_emotion is not None and hse_conf >= HSEMOTION_CONFIDENCE_THRESHOLD:
        return hse_emotion, float(hse_conf), "hsemotion", False

    if df_emotion is not None and df_conf >= DEEPFACE_CONFIDENCE_THRESHOLD:
        return df_emotion, float(df_conf), "deepface", False

    if (df_conf > hse_conf) and df_emotion is not None:
        return df_emotion, float(df_conf), "deepface", False
    if hse_emotion is not None:
        return hse_emotion, float(hse_conf), "hsemotion", False

    return None, 0.0, "none", True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@dataclass
class Task:
    rel_dir: str
    rel_path: str
    filename: str
    frame_index: int
    face_id: int
    track_id: int
    identity_id: int
    image_path: str


def analyze_emotions_incremental(
    faces_root: str,
    output_root: str,
    master_json_path: str,
):
    """
    Analyse incrémentale:
    - lit faces_root (images crops)
    - calcule HSEmotion + DeepFace (TTA optionnel)
    - ajoute métriques qualité
    - smoothing centré (n-1,n,n+1) par rel_dir + identity_id (2e passe)
    - garde un JSON master global + outputs par sous-dossier
    - écrit 2 JSON:
        1) emotions.json complet
        2) emotions_final.json (résultats finaux uniquement)
    """

    os.makedirs(output_root, exist_ok=True)
    master_results = load_master_json(master_json_path)

    # ---------------- GPU selection ----------------
    # HSEmotion sur GPU si dispo
    hse_device = _select_hse_device("cuda")
    hse_detector = HSEmotionDetector(device=hse_device)

    # DeepFace: TF GPU si dispo (config plus haut), sinon CPU sans casser
    df_detector = DeepFaceEmotionDetector(
        detector_backend=DEEPFACE_DETECTOR_BACKEND,
        enforce_detection=DEEPFACE_ENFORCE_DETECTION,
    )

    tasks: List[Task] = []
    for dirpath, _, filenames in os.walk(faces_root):
        rel_dir = os.path.relpath(dirpath, faces_root)

        for filename in filenames:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)
            if rel_path in master_results:
                continue

            frame_index = parse_frame_index(filename)
            if frame_index == -1:
                continue

            fid = parse_face_id(filename)
            tid = parse_track_id(filename)
            iid = identity_id(fid, tid)

            image_path = os.path.join(dirpath, filename)

            tasks.append(Task(
                rel_dir=rel_dir,
                rel_path=rel_path,
                filename=filename,
                frame_index=frame_index,
                face_id=fid,
                track_id=tid,
                identity_id=iid,
                image_path=image_path,
            ))

    if not tasks:
        print("Aucune nouvelle image à analyser. Tout est déjà à jour ✅")
        return

    tasks.sort(key=lambda t: (t.rel_dir, t.identity_id, t.frame_index, t.filename))

    per_dir_rows: Dict[str, List[dict]] = {}
    per_dir_json: Dict[str, Dict[str, dict]] = {}

    for t in tasks:
        image = cv2.imread(t.image_path)
        if image is None:
            print(f"Erreur: impossible de lire {t.image_path}")
            continue

        h, w = image.shape[:2]
        if min(h, w) < MIN_FACE_SIZE:
            q = face_quality_metrics(image)
            entry = {
                "relative_path": t.rel_path,
                "filename": t.filename,
                "frame_index": t.frame_index,
                "face_id": t.face_id,
                "track_id": t.track_id,
                "identity_id": t.identity_id,

                "hse_emotion": None,
                "hse_confidence": 0.0,
                "deepface_emotion": None,
                "deepface_confidence": 0.0,

                "agree": False,

                "final_emotion": None,
                "final_confidence": 0.0,
                "final_backend": "too_small",
                "is_uncertain": True,

                # champs smoothing (remplis en 2e passe)
                "smoothed_hse_emotion": None,
                "smoothed_deepface_emotion": None,
                "smoothed_final_emotion": None,

                "quality_blur": q["blur"],
                "quality_brightness": q["brightness"],
                "quality_contrast": q["contrast"],
                "quality_area": q["area"],
                "bad_quality": True,
            }
            per_dir_rows.setdefault(t.rel_dir, []).append(entry)
            per_dir_json.setdefault(t.rel_dir, {})[t.rel_path] = entry
            master_results[t.rel_path] = entry
            continue

        q = face_quality_metrics(image)
        bad_q = is_bad_quality(q)

        # --- HSEmotion (GPU si dispo) ---
        hse_emotion, hse_conf = hse_detector.analyze(image, use_tta=ENABLE_TTA)

        # --- DeepFace (TF GPU si dispo, sinon CPU) ---
        df_emotion, df_conf = (None, 0.0)
        if RUN_DEEPFACE_ALWAYS or (hse_emotion is None or hse_conf < HSEMOTION_CONFIDENCE_THRESHOLD):
            df_emotion, df_conf = df_detector.analyze(image, use_tta=ENABLE_TTA)

        agree = (
            hse_emotion is not None
            and df_emotion is not None
            and hse_emotion == df_emotion
        )

        final_emotion, final_conf, final_backend, is_uncertain = decide_final(
            hse_emotion, hse_conf, df_emotion, df_conf, bad_q
        )

        entry = {
            "relative_path": t.rel_path,
            "filename": t.filename,
            "frame_index": t.frame_index,
            "face_id": t.face_id,
            "track_id": t.track_id,
            "identity_id": t.identity_id,

            "hse_emotion": hse_emotion,
            "hse_confidence": float(hse_conf),
            "deepface_emotion": df_emotion,
            "deepface_confidence": float(df_conf),
            "agree": bool(agree),

            "final_emotion": final_emotion,
            "final_confidence": float(final_conf),
            "final_backend": final_backend,
            "is_uncertain": bool(is_uncertain),

            # champs smoothing (remplis en 2e passe)
            "smoothed_hse_emotion": None,
            "smoothed_deepface_emotion": None,
            "smoothed_final_emotion": None,

            "quality_blur": q["blur"],
            "quality_brightness": q["brightness"],
            "quality_contrast": q["contrast"],
            "quality_area": q["area"],
            "bad_quality": bool(bad_q),
        }

        per_dir_rows.setdefault(t.rel_dir, []).append(entry)
        per_dir_json.setdefault(t.rel_dir, {})[t.rel_path] = entry
        master_results[t.rel_path] = entry

    # -------------------------------------------------------------------------
    # 2e passe: smoothing centré sur per_dir_rows (n-1,n,n+1)
    # -------------------------------------------------------------------------
    if ENABLE_SMOOTHING:
        apply_centered_smoothing_per_dir(per_dir_rows, window=CENTERED_SMOOTH_WINDOW)

        # synchroniser aussi per_dir_json
        for rel_dir, rows in per_dir_rows.items():
            for r in rows:
                rp = r["relative_path"]
                if rel_dir in per_dir_json and rp in per_dir_json[rel_dir]:
                    per_dir_json[rel_dir][rp] = r

        # synchroniser master_results (pour les nouveaux)
        for rel_dir, rows in per_dir_rows.items():
            for r in rows:
                rp = r["relative_path"]
                master_results[rp] = r

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    fieldnames = [
        "relative_path",
        "filename",
        "frame_index",
        "face_id",
        "track_id",
        "identity_id",

        "hse_emotion",
        "hse_confidence",
        "deepface_emotion",
        "deepface_confidence",
        "agree",

        "final_emotion",
        "final_confidence",
        "final_backend",
        "is_uncertain",

        "smoothed_hse_emotion",
        "smoothed_deepface_emotion",
        "smoothed_final_emotion",

        "quality_blur",
        "quality_brightness",
        "quality_contrast",
        "quality_area",
        "bad_quality",
    ]

    for rel_dir, rows in per_dir_rows.items():
        base_dir = "root" if rel_dir == "." else rel_dir
        run_folder = os.path.join(output_root, base_dir, timestamp)
        os.makedirs(run_folder, exist_ok=True)

        csv_path = os.path.join(run_folder, "emotions.csv")
        json_path = os.path.join(run_folder, "emotions.json")
        final_json_path = os.path.join(run_folder, "emotions_final.json")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            csvfile.write(f"# Emotions analysis run at {timestamp}\n")
            csvfile.write(f"# DeepFace backend: {DEEPFACE_DETECTOR_BACKEND}, enforce={DEEPFACE_ENFORCE_DETECTION}\n")
            csvfile.write(f"# TTA: {ENABLE_TTA}, smoothing: {ENABLE_SMOOTHING} (centered window={CENTERED_SMOOTH_WINDOW})\n")
            csvfile.write(f"# Fallback thresholds: HSE={HSEMOTION_CONFIDENCE_THRESHOLD}, DF={DEEPFACE_CONFIDENCE_THRESHOLD}\n")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        # JSON complet
        with open(json_path, mode="w", encoding="utf-8") as jsonfile:
            json.dump(per_dir_json[rel_dir], jsonfile, indent=4, ensure_ascii=False)

        # JSON "final only"
        final_only = {}
        for row in rows:
            rp = row["relative_path"]
            final_only[rp] = {
                "final_emotion": row.get("final_emotion"),
                "final_confidence": row.get("final_confidence", 0.0),
                "final_backend": row.get("final_backend"),
                "is_uncertain": row.get("is_uncertain", True),
                "smoothed_final_emotion": row.get("smoothed_final_emotion"),
                "frame_index": row.get("frame_index"),
                "identity_id": row.get("identity_id"),
                "track_id": row.get("track_id"),
                "face_id": row.get("face_id"),
            }

        with open(final_json_path, mode="w", encoding="utf-8") as f:
            json.dump(final_only, f, indent=4, ensure_ascii=False)

        print(f"{len(rows)} faces analysées pour le dossier '{rel_dir}' ✅")
        print(f"→ CSV :        {csv_path}")
        print(f"→ JSON :       {json_path}")
        print(f"→ JSON final : {final_json_path}")

    save_master_json(master_results, master_json_path)
    print(f"Master JSON mis à jour : {master_json_path}")


if __name__ == "__main__":
    faces_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces"
    output_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\output\\emotions"
    master_json_path = os.path.join(output_root, "emotions_master.json")

    analyze_emotions_incremental(faces_root, output_root, master_json_path)
