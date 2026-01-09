# # =========================
# # analyze_emotion.py
# # =========================
# import os
# import re
# import cv2
# import csv
# import json
# import traceback
# import argparse
# import numpy as np
# from dataclasses import dataclass
# from datetime import datetime
# from collections import Counter
# from typing import Dict, List, Optional, Tuple
# from pathlib import Path

# from deepface import DeepFace

# # =============================================================================
# # CONFIG
# # =============================================================================

# MIN_FACE_SIZE = 60

# BLUR_MIN_VAR_LAPLACIAN = 60.0
# BRIGHTNESS_MIN = 35.0
# BRIGHTNESS_MAX = 220.0

# DEEPFACE_DETECTOR_BACKEND = "skip"
# DEEPFACE_ENFORCE_DETECTION = True

# ENABLE_TTA = True
# TTA_ROT_DEG = 5
# TTA_BRIGHT_FACTOR = 0.10
# TTA_MAX_VARIANTS = 5

# ENABLE_SMOOTHING = True
# CENTERED_SMOOTH_WINDOW = 3

# ENABLE_SMOOTHING_SCORE = True
# SMOOTH_SCORE_USE_FINAL = True

# RUN_DEEPFACE_ALWAYS = False
# HSEMOTION_CONFIDENCE_THRESHOLD = 0.65
# DEEPFACE_CONFIDENCE_THRESHOLD = 0.70

# ENABLE_UNCERTAIN_CLASS = True
# UNCERTAIN_MIN_CONF = 0.55


# # =============================================================================
# # UTIL PARSING
# # =============================================================================

# def parse_frame_index(filename: str) -> int:
#     m = re.search(r"frame_(\d+)", filename.lower())
#     return int(m.group(1)) if m else -1

# def parse_face_id(filename: str) -> int:
#     try:
#         lower = filename.lower()
#         for key in ("track", "face"):
#             idx = lower.rfind(key)
#             if idx != -1:
#                 tail = lower[idx + len(key):]
#                 digits = ""
#                 for ch in tail:
#                     if ch.isdigit():
#                         digits += ch
#                     else:
#                         break
#                 return int(digits) if digits else -1
#         return -1
#     except Exception:
#         return -1

# def parse_track_id(filename: str) -> int:
#     try:
#         lower = filename.lower()
#         idx = lower.rfind("track")
#         if idx == -1:
#             return -1
#         tail = lower[idx + 5:]
#         digits = ""
#         for ch in tail:
#             if ch.isdigit():
#                 digits += ch
#             else:
#                 break
#         return int(digits) if digits else -1
#     except Exception:
#         return -1

# def identity_id(face_id: int, track_id: int) -> int:
#     return track_id if track_id != -1 else face_id


# # =============================================================================
# # GLOBAL PERSON ID (UNIQUE ACROSS VIDEOS)
# # =============================================================================

# def _norm_key(s: str) -> str:
#     """Normalise en clé stable: slash/backslash/espaces -> '_'."""
#     s = (s or "").replace("\\", "/").strip("/")
#     s = re.sub(r"\s+", "_", s)
#     s = s.replace("/", "_")
#     return s

# def extract_person_folder(rel_dir: str) -> Optional[str]:
#     """
#     Cherche un segment 'person_0000' dans rel_dir.
#     Retourne 'person_0000' si trouvé, sinon None.
#     """
#     if not rel_dir:
#         return None
#     parts = rel_dir.replace("\\", "/").split("/")
#     for p in parts[::-1]:
#         if re.fullmatch(r"person[_-]?\d+", p, flags=re.IGNORECASE):
#             num = re.search(r"(\d+)", p).group(1)
#             return f"person_{int(num):04d}"
#     return None

# def extract_video_key(rel_dir: str) -> str:
#     """
#     Déduit la 'vidéo' à partir de rel_dir en supprimant person_XXXX.
#     Exemple:
#       bedouk/frames_.../person_0000  -> bedouk_frames_...
#     """
#     if not rel_dir:
#         return "unknown_video"
#     parts = [p for p in rel_dir.replace("\\", "/").split("/") if p]
#     parts = [p for p in parts if not re.fullmatch(r"person[_-]?\d+", p, flags=re.IGNORECASE)]
#     key = "_".join(parts) if parts else "unknown_video"
#     return _norm_key(key)

# def make_global_person_id(rel_dir: str, identity_id_int: int) -> str:
#     """
#     Construit un identifiant global unique:
#       <video_key>_<person_XXXX>
#     - si rel_dir contient person_XXXX -> on l’utilise
#     - sinon fallback -> identity_id -> person_XXXX
#     """
#     video_key = extract_video_key(rel_dir)
#     person_folder = extract_person_folder(rel_dir)
#     if person_folder is None:
#         person_folder = f"person_{max(0, int(identity_id_int)):04d}"
#     return f"{video_key}_{person_folder}"


# # =============================================================================
# # MASTER JSON
# # =============================================================================

# def load_master_json(master_json_path: str) -> dict:
#     if os.path.exists(master_json_path):
#         with open(master_json_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {}

# def save_master_json(master: dict, master_json_path: str):
#     os.makedirs(os.path.dirname(master_json_path), exist_ok=True)
#     with open(master_json_path, "w", encoding="utf-8") as f:
#         json.dump(master, f, indent=4, ensure_ascii=False)


# # =============================================================================
# # QUALITY METRICS
# # =============================================================================

# def face_quality_metrics(face_bgr: np.ndarray) -> Dict[str, float]:
#     if face_bgr is None or face_bgr.size == 0:
#         return {"blur": 0.0, "brightness": 0.0, "contrast": 0.0, "area": 0.0}

#     gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
#     blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
#     brightness = float(np.mean(gray))
#     contrast = float(np.std(gray))
#     area = float(face_bgr.shape[0] * face_bgr.shape[1])
#     return {"blur": blur, "brightness": brightness, "contrast": contrast, "area": area}

# def is_bad_quality(q: Dict[str, float]) -> bool:
#     if q["area"] <= 0:
#         return True
#     if q["blur"] < BLUR_MIN_VAR_LAPLACIAN:
#         return True
#     if q["brightness"] < BRIGHTNESS_MIN or q["brightness"] > BRIGHTNESS_MAX:
#         return True
#     return False


# # =============================================================================
# # TTA
# # =============================================================================

# def tta_variants(face_bgr: np.ndarray) -> List[np.ndarray]:
#     variants = [face_bgr]
#     if face_bgr is None or face_bgr.size == 0:
#         return variants

#     variants.append(cv2.flip(face_bgr, 1))

#     img = face_bgr.astype(np.float32)
#     brighter = np.clip(img * (1.0 + TTA_BRIGHT_FACTOR), 0, 255).astype(np.uint8)
#     darker = np.clip(img * (1.0 - TTA_BRIGHT_FACTOR), 0, 255).astype(np.uint8)
#     variants.append(brighter)
#     variants.append(darker)

#     h, w = face_bgr.shape[:2]
#     M = cv2.getRotationMatrix2D((w / 2, h / 2), TTA_ROT_DEG, 1.0)
#     rot = cv2.warpAffine(
#         face_bgr, M, (w, h),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REFLECT
#     )
#     variants.append(rot)

#     return variants[:TTA_MAX_VARIANTS]


# # =============================================================================
# # SMOOTHING + SCORE
# # =============================================================================

# def centered_mode(values: List[Optional[str]]) -> Optional[str]:
#     vals = [v for v in values if v is not None]
#     if not vals:
#         return None
#     return Counter(vals).most_common(1)[0][0]

# def _safe_float(x, default=0.0) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return default

# def compute_smoothing_score(
#     emotions_window: List[Optional[str]],
#     confidences_window: List[float],
# ) -> Tuple[Optional[str], float, float, float]:
#     emo = [e for e in emotions_window if isinstance(e, str) and e.strip()]
#     if not emo:
#         return None, 0.0, 0.0, 0.0

#     counts = Counter(emo)
#     smoothed_emotion, top_count = counts.most_common(1)[0]
#     total = len(emo)
#     vote_ratio = (top_count / total) if total else 0.0

#     confs = [max(0.0, min(1.0, _safe_float(c, 0.0))) for c in confidences_window]
#     avg_conf = (sum(confs) / len(confs)) if confs else 0.0

#     smoothing_score = vote_ratio * avg_conf
#     return smoothed_emotion, smoothing_score, vote_ratio, avg_conf

# def apply_centered_smoothing(entries: List[dict], window: int, key_emotion: str, out_key: str):
#     if window % 2 != 1 or window < 3:
#         raise ValueError("window doit être impair et >= 3")

#     half = window // 2
#     n = len(entries)

#     for i in range(n):
#         lo = max(0, i - half)
#         hi = min(n, i + half + 1)
#         vals = [entries[j].get(key_emotion) for j in range(lo, hi)]
#         entries[i][out_key] = centered_mode(vals)

# def apply_smoothing_score_on_key(
#     entries: List[dict],
#     window: int,
#     emotion_key: str,
#     conf_key: str,
#     out_score_key: str,
#     out_vote_key: str,
#     out_avgconf_key: str,
#     out_winsize_key: str,
# ):
#     if window % 2 != 1 or window < 3:
#         raise ValueError("window doit être impair et >= 3")

#     half = window // 2
#     n = len(entries)

#     for i in range(n):
#         lo = max(0, i - half)
#         hi = min(n, i + half + 1)

#         emos = [entries[j].get(emotion_key) for j in range(lo, hi)]
#         confs = [entries[j].get(conf_key, 0.0) for j in range(lo, hi)]

#         _, score, vote_ratio, avg_conf = compute_smoothing_score(emos, confs)

#         entries[i][out_score_key] = float(score)
#         entries[i][out_vote_key] = float(vote_ratio)
#         entries[i][out_avgconf_key] = float(avg_conf)
#         entries[i][out_winsize_key] = int(hi - lo)

# def apply_centered_smoothing_per_dir(per_dir_rows: Dict[str, List[dict]], window: int):
#     for rel_dir, rows in per_dir_rows.items():
#         groups: Dict[int, List[dict]] = {}
#         for r in rows:
#             iid = int(r.get("identity_id", -1))
#             groups.setdefault(iid, []).append(r)

#         for _, lst in groups.items():
#             lst.sort(key=lambda x: int(x.get("frame_index", -1)))

#             apply_centered_smoothing(lst, window=window, key_emotion="hse_emotion", out_key="smoothed_hse_emotion")
#             apply_centered_smoothing(lst, window=window, key_emotion="deepface_emotion", out_key="smoothed_deepface_emotion")

#             for r in lst:
#                 sh = r.get("smoothed_hse_emotion")
#                 sd = r.get("smoothed_deepface_emotion")
#                 if sh is not None and sd is not None:
#                     r["smoothed_final_emotion"] = sh if sh == sd else sh
#                 else:
#                     r["smoothed_final_emotion"] = sh or sd

#             if ENABLE_SMOOTHING_SCORE:
#                 if SMOOTH_SCORE_USE_FINAL:
#                     apply_smoothing_score_on_key(
#                         lst, window=window,
#                         emotion_key="final_emotion",
#                         conf_key="final_confidence",
#                         out_score_key="smoothing_score",
#                         out_vote_key="smoothing_vote_ratio",
#                         out_avgconf_key="smoothing_avg_conf_window",
#                         out_winsize_key="smoothing_window_size",
#                     )
#                 else:
#                     apply_smoothing_score_on_key(
#                         lst, window=window,
#                         emotion_key="smoothed_final_emotion",
#                         conf_key="final_confidence",
#                         out_score_key="smoothing_score",
#                         out_vote_key="smoothing_vote_ratio",
#                         out_avgconf_key="smoothing_avg_conf_window",
#                         out_winsize_key="smoothing_window_size",
#                     )

#                 for r in lst:
#                     r["was_smoothed_changed"] = (r.get("smoothed_final_emotion") != r.get("final_emotion"))


# # =============================================================================
# # BACKENDS
# # =============================================================================

# class DeepFaceEmotionDetector:
#     def __init__(self, detector_backend: str = DEEPFACE_DETECTOR_BACKEND, enforce_detection: bool = DEEPFACE_ENFORCE_DETECTION):
#         self.detector_backend = detector_backend
#         self.enforce_detection = enforce_detection
#         print(f"[DeepFaceEmotionDetector] Ready (backend={detector_backend}, enforce={enforce_detection})")

#     def analyze_once(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
#         if img_bgr is None or img_bgr.size == 0:
#             return None, 0.0

#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         pred = DeepFace.analyze(
#             img_rgb,
#             actions=["emotion"],
#             enforce_detection=self.enforce_detection,
#             detector_backend=self.detector_backend,
#             silent=True,
#         )

#         res0 = pred[0] if isinstance(pred, list) and pred else pred
#         if not isinstance(res0, dict):
#             return None, 0.0

#         dominant = res0.get("dominant_emotion", None)
#         scores = res0.get("emotion", {}) or {}
#         raw = float(scores.get(dominant, 0.0)) if dominant else 0.0
#         conf = raw / 100.0 if raw > 1.5 else raw
#         return dominant, conf

#     def analyze(self, img_bgr: np.ndarray, use_tta: bool = False) -> Tuple[Optional[str], float]:
#         if not use_tta:
#             try:
#                 return self.analyze_once(img_bgr)
#             except Exception:
#                 return None, 0.0

#         emotions, confs = [], []
#         for v in tta_variants(img_bgr):
#             try:
#                 e, c = self.analyze_once(v)
#                 if e is not None:
#                     emotions.append(e)
#                     confs.append(float(c))
#             except Exception:
#                 continue

#         if not emotions:
#             return None, 0.0

#         top = Counter(emotions).most_common(1)[0][0]
#         mean_conf = float(np.mean(confs)) if confs else 0.0
#         return top, mean_conf


# class HSEmotionDetector:
#     def __init__(self, device: str = "cpu"):
#         self._printed_error = False
#         print("[HSEmotionDetector] Chargement du modèle HSEmotion...")
#         from hsemotion.facial_emotions import HSEmotionRecognizer
#         self.model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf", device=device)
#         print("[HSEmotionDetector] Modèle chargé ✅")

#     def analyze_once(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
#         if img_bgr is None or img_bgr.size == 0:
#             return None, 0.0
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         emotion, scores = self.model.predict_emotions(img_rgb, logits=False)
#         conf = float(np.max(scores)) if scores is not None else 0.0
#         return emotion, conf

#     def analyze(self, img_bgr: np.ndarray, use_tta: bool = False) -> Tuple[Optional[str], float]:
#         if not use_tta:
#             try:
#                 return self.analyze_once(img_bgr)
#             except Exception:
#                 if not self._printed_error:
#                     self._printed_error = True
#                     print("[HSEmotionDetector] ERREUR (une seule fois) :")
#                     traceback.print_exc()
#                 return None, 0.0

#         emotions, confs = [], []
#         for v in tta_variants(img_bgr):
#             try:
#                 e, c = self.analyze_once(v)
#                 if e is not None:
#                     emotions.append(e)
#                     confs.append(float(c))
#             except Exception:
#                 continue

#         if not emotions:
#             return None, 0.0

#         top = Counter(emotions).most_common(1)[0][0]
#         mean_conf = float(np.mean(confs)) if confs else 0.0
#         return top, mean_conf


# # =============================================================================
# # DECISION LOGIC
# # =============================================================================

# def decide_final(hse_emotion, hse_conf, df_emotion, df_conf, quality_bad) -> Tuple[Optional[str], float, str, bool]:
#     if ENABLE_UNCERTAIN_CLASS:
#         low_both = (hse_conf < UNCERTAIN_MIN_CONF) and (df_conf < UNCERTAIN_MIN_CONF)
#         disagree = (hse_emotion is not None and df_emotion is not None and hse_emotion != df_emotion)
#         if low_both or (quality_bad and disagree):
#             return None, 0.0, "uncertain", True

#     if hse_emotion is not None and hse_conf >= HSEMOTION_CONFIDENCE_THRESHOLD:
#         return hse_emotion, float(hse_conf), "hsemotion", False

#     if df_emotion is not None and df_conf >= DEEPFACE_CONFIDENCE_THRESHOLD:
#         return df_emotion, float(df_conf), "deepface", False

#     if (df_conf > hse_conf) and df_emotion is not None:
#         return df_emotion, float(df_conf), "deepface", False
#     if hse_emotion is not None:
#         return hse_emotion, float(hse_conf), "hsemotion", False

#     return None, 0.0, "none", True


# # =============================================================================
# # SKIP HELPERS (MASTER-BASED)
# # =============================================================================

# def dir_fully_processed(rel_dir: str, img_files: List[str], master_results: dict) -> bool:
#     """
#     ✅ True si toutes les images de rel_dir sont déjà dans emotion_results_master.json
#     """
#     if not img_files:
#         return False
#     for fn in img_files:
#         rel_path = fn if rel_dir == "." else os.path.join(rel_dir, fn)
#         if rel_path not in master_results:
#             return False
#     return True


# # =============================================================================
# # MAIN PIPELINE
# # =============================================================================

# @dataclass
# class Task:
#     rel_dir: str
#     rel_path: str
#     filename: str
#     frame_index: int
#     face_id: int
#     track_id: int
#     identity_id: int
#     global_person_id: str
#     image_path: str


# def analyze_emotions_incremental(faces_root: str, output_root: str, master_json_path: str):
#     """
#     - Source: faces_root = data/detected_faces
#     - Skip par dossier: si toutes les images du dossier sont déjà dans emotion_results_master.json
#     - Output stable: output_root/<rel_dir>/latest/ (pas de timestamp)
#     - Incremental: si dossier partiel, analyse seulement les nouvelles images
#     """

#     os.makedirs(output_root, exist_ok=True)
#     master_results = load_master_json(master_json_path)

#     hse_detector = HSEmotionDetector(device="cpu")
#     df_detector = DeepFaceEmotionDetector(
#         detector_backend=DEEPFACE_DETECTOR_BACKEND,
#         enforce_detection=DEEPFACE_ENFORCE_DETECTION,
#     )

#     tasks: List[Task] = []

#     for dirpath, _, filenames in os.walk(faces_root):
#         rel_dir = os.path.relpath(dirpath, faces_root)

#         img_files = [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#         if not img_files:
#             continue

#         # ✅ SKIP si dossier complet déjà traité
#         if dir_fully_processed(rel_dir, img_files, master_results):
#             print(f"[SKIP] Dossier déjà analysé (master complet): {rel_dir}")
#             continue

#         for filename in img_files:
#             rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)
#             if rel_path in master_results:
#                 continue

#             frame_index = parse_frame_index(filename)
#             if frame_index == -1:
#                 continue

#             fid = parse_face_id(filename)
#             tid = parse_track_id(filename)
#             iid = identity_id(fid, tid)
#             gid = make_global_person_id(rel_dir, iid)

#             image_path = os.path.join(dirpath, filename)

#             tasks.append(Task(
#                 rel_dir=rel_dir,
#                 rel_path=rel_path,
#                 filename=filename,
#                 frame_index=frame_index,
#                 face_id=fid,
#                 track_id=tid,
#                 identity_id=iid,
#                 global_person_id=gid,
#                 image_path=image_path,
#             ))

#     if not tasks:
#         print("Aucune nouvelle image à analyser. Tout est déjà à jour ✅")
#         return

#     tasks.sort(key=lambda t: (t.rel_dir, t.identity_id, t.frame_index, t.filename))

#     per_dir_rows: Dict[str, List[dict]] = {}
#     per_dir_json: Dict[str, Dict[str, dict]] = {}

#     for t in tasks:
#         image = cv2.imread(t.image_path)
#         if image is None:
#             print(f"Erreur: impossible de lire {t.image_path}")
#             continue

#         h, w = image.shape[:2]
#         if min(h, w) < MIN_FACE_SIZE:
#             q = face_quality_metrics(image)
#             entry = {
#                 "relative_path": t.rel_path,
#                 "filename": t.filename,
#                 "frame_index": t.frame_index,
#                 "face_id": t.face_id,
#                 "track_id": t.track_id,
#                 "identity_id": t.identity_id,
#                 "global_person_id": t.global_person_id,

#                 "hse_emotion": None,
#                 "hse_confidence": 0.0,
#                 "deepface_emotion": None,
#                 "deepface_confidence": 0.0,
#                 "agree": False,

#                 "final_emotion": None,
#                 "final_confidence": 0.0,
#                 "final_backend": "too_small",
#                 "is_uncertain": True,

#                 "smoothed_hse_emotion": None,
#                 "smoothed_deepface_emotion": None,
#                 "smoothed_final_emotion": None,

#                 "smoothing_score": 0.0,
#                 "smoothing_vote_ratio": 0.0,
#                 "smoothing_avg_conf_window": 0.0,
#                 "smoothing_window_size": 0,
#                 "was_smoothed_changed": False,

#                 "quality_blur": q["blur"],
#                 "quality_brightness": q["brightness"],
#                 "quality_contrast": q["contrast"],
#                 "quality_area": q["area"],
#                 "bad_quality": True,
#             }
#             per_dir_rows.setdefault(t.rel_dir, []).append(entry)
#             per_dir_json.setdefault(t.rel_dir, {})[t.rel_path] = entry
#             master_results[t.rel_path] = entry
#             continue

#         q = face_quality_metrics(image)
#         bad_q = is_bad_quality(q)

#         hse_emotion, hse_conf = hse_detector.analyze(image, use_tta=ENABLE_TTA)

#         df_emotion, df_conf = (None, 0.0)
#         if RUN_DEEPFACE_ALWAYS or (hse_emotion is None or hse_conf < HSEMOTION_CONFIDENCE_THRESHOLD):
#             df_emotion, df_conf = df_detector.analyze(image, use_tta=ENABLE_TTA)

#         agree = (
#             hse_emotion is not None
#             and df_emotion is not None
#             and hse_emotion == df_emotion
#         )

#         final_emotion, final_conf, final_backend, is_uncertain = decide_final(
#             hse_emotion, hse_conf, df_emotion, df_conf, bad_q
#         )

#         entry = {
#             "relative_path": t.rel_path,
#             "filename": t.filename,
#             "frame_index": t.frame_index,
#             "face_id": t.face_id,
#             "track_id": t.track_id,
#             "identity_id": t.identity_id,
#             "global_person_id": t.global_person_id,

#             "hse_emotion": hse_emotion,
#             "hse_confidence": float(hse_conf),
#             "deepface_emotion": df_emotion,
#             "deepface_confidence": float(df_conf),
#             "agree": bool(agree),

#             "final_emotion": final_emotion,
#             "final_confidence": float(final_conf),
#             "final_backend": final_backend,
#             "is_uncertain": bool(is_uncertain),

#             "smoothed_hse_emotion": None,
#             "smoothed_deepface_emotion": None,
#             "smoothed_final_emotion": None,

#             "smoothing_score": 0.0,
#             "smoothing_vote_ratio": 0.0,
#             "smoothing_avg_conf_window": 0.0,
#             "smoothing_window_size": 0,
#             "was_smoothed_changed": False,

#             "quality_blur": q["blur"],
#             "quality_brightness": q["brightness"],
#             "quality_contrast": q["contrast"],
#             "quality_area": q["area"],
#             "bad_quality": bool(bad_q),
#         }

#         per_dir_rows.setdefault(t.rel_dir, []).append(entry)
#         per_dir_json.setdefault(t.rel_dir, {})[t.rel_path] = entry
#         master_results[t.rel_path] = entry

#     # 2e passe: smoothing + score
#     if ENABLE_SMOOTHING:
#         apply_centered_smoothing_per_dir(per_dir_rows, window=CENTERED_SMOOTH_WINDOW)

#         # sync json + master
#         for rel_dir, rows in per_dir_rows.items():
#             for r in rows:
#                 rp = r["relative_path"]
#                 if rel_dir in per_dir_json and rp in per_dir_json[rel_dir]:
#                     per_dir_json[rel_dir][rp] = r
#                 master_results[rp] = r

#     timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

#     fieldnames = [
#         "relative_path","filename","frame_index","face_id","track_id","identity_id","global_person_id",
#         "hse_emotion","hse_confidence","deepface_emotion","deepface_confidence","agree",
#         "final_emotion","final_confidence","final_backend","is_uncertain",
#         "smoothed_hse_emotion","smoothed_deepface_emotion","smoothed_final_emotion",
#         "smoothing_score","smoothing_vote_ratio","smoothing_avg_conf_window","smoothing_window_size","was_smoothed_changed",
#         "quality_blur","quality_brightness","quality_contrast","quality_area","bad_quality",
#     ]

#     for rel_dir, rows in per_dir_rows.items():
#         base_dir = "root" if rel_dir == "." else rel_dir

#         # ✅ dossier stable (pas de timestamp)
#         run_folder = os.path.join(output_root, base_dir, "latest")
#         os.makedirs(run_folder, exist_ok=True)

#         csv_path = os.path.join(run_folder, "analyzed_emotions.csv")
#         json_path = os.path.join(run_folder, "analyzed_emotions.json")
#         final_json_path = os.path.join(run_folder, "analyzed_emotions_final.json")

#         with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
#             csvfile.write(f"# Emotions analysis run at {timestamp}\n")
#             csvfile.write(f"# DeepFace backend: {DEEPFACE_DETECTOR_BACKEND}, enforce={DEEPFACE_ENFORCE_DETECTION}\n")
#             csvfile.write(f"# TTA: {ENABLE_TTA}, smoothing: {ENABLE_SMOOTHING} (window={CENTERED_SMOOTH_WINDOW})\n")
#             csvfile.write(f"# Fallback thresholds: HSE={HSEMOTION_CONFIDENCE_THRESHOLD}, DF={DEEPFACE_CONFIDENCE_THRESHOLD}\n")
#             csvfile.write(f"# Smoothing score: {ENABLE_SMOOTHING_SCORE} (use_final={SMOOTH_SCORE_USE_FINAL})\n")
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             for row in rows:
#                 writer.writerow(row)

#         with open(json_path, mode="w", encoding="utf-8") as jsonfile:
#             json.dump(per_dir_json[rel_dir], jsonfile, indent=4, ensure_ascii=False)

#         final_only = {}
#         for row in rows:
#             rp = row["relative_path"]
#             final_only[rp] = {
#                 "final_emotion": row.get("final_emotion"),
#                 "final_confidence": row.get("final_confidence", 0.0),
#                 "final_backend": row.get("final_backend"),
#                 "is_uncertain": row.get("is_uncertain", True),
#                 "smoothed_final_emotion": row.get("smoothed_final_emotion"),
#                 "smoothing_score": row.get("smoothing_score", 0.0),
#                 "smoothing_vote_ratio": row.get("smoothing_vote_ratio", 0.0),
#                 "smoothing_avg_conf_window": row.get("smoothing_avg_conf_window", 0.0),
#                 "smoothing_window_size": row.get("smoothing_window_size", 0),
#                 "was_smoothed_changed": row.get("was_smoothed_changed", False),
#                 "frame_index": row.get("frame_index"),
#                 "identity_id": row.get("identity_id"),
#                 "global_person_id": row.get("global_person_id"),
#                 "track_id": row.get("track_id"),
#                 "face_id": row.get("face_id"),
#             }

#         with open(final_json_path, mode="w", encoding="utf-8") as f:
#             json.dump(final_only, f, indent=4, ensure_ascii=False)

#         print(f"{len(rows)} faces analysées pour le dossier '{rel_dir}' ✅")
#         print(f"→ CSV :        {csv_path}")
#         print(f"→ JSON :       {json_path}")
#         print(f"→ JSON final : {final_json_path}")

#     save_master_json(master_results, master_json_path)
#     print(f"Master JSON mis à jour : {master_json_path}")


# # =============================================================================
# # CLI
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(description="Analyze emotions from detected faces (VideoEmotion)")
#     parser.add_argument(
#         "--faces-root",
#         default=None,
#         help="Dossier racine des faces détectées (défaut: data/detected_faces).",
#     )
#     parser.add_argument(
#         "--output-root",
#         default=None,
#         help="Dossier racine de sortie (défaut: output/emotion_results).",
#     )
#     parser.add_argument(
#         "--master-json",
#         default=None,
#         help="Chemin vers le master JSON (défaut: <output-root>/emotion_results_master.json).",
#     )
#     parser.add_argument(
#         "--project-root",
#         default=None,
#         help="Racine du projet (défaut: auto).",
#     )
#     args = parser.parse_args()

#     project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]

#     faces_root = Path(args.faces_root).resolve() if args.faces_root else (project_root / "data" / "detected_faces")
#     output_root = Path(args.output_root).resolve() if args.output_root else (project_root / "output" / "emotion_results")

#     if args.master_json:
#         master_json_path = Path(args.master_json)
#         if not master_json_path.is_absolute():
#             master_json_path = (project_root / master_json_path).resolve()
#     else:
#         master_json_path = (output_root / "emotion_results_master.json").resolve()

#     if not faces_root.exists():
#         print(f"[ERREUR] Dossier faces introuvable: {faces_root}")
#         return

#     output_root.mkdir(parents=True, exist_ok=True)

#     analyze_emotions_incremental(
#         faces_root=str(faces_root),
#         output_root=str(output_root),
#         master_json_path=str(master_json_path),
#     )


# if __name__ == "__main__":
#     main()

# #commande to run the pipeline
# # python src/analyze_emotion.py --faces-root data/detected_faces --output-root output/emotion_results

# =========================
# analyze_emotion.py
# =========================
import os
import re
import cv2
import csv
import json
import traceback
import argparse
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import yaml
from deepface import DeepFace

# =============================================================================
# CONFIG (defaults, override possible via config.yaml)
# =============================================================================

MIN_FACE_SIZE = 60

BLUR_MIN_VAR_LAPLACIAN = 60.0
BRIGHTNESS_MIN = 35.0
BRIGHTNESS_MAX = 220.0

DEEPFACE_DETECTOR_BACKEND = "skip"
DEEPFACE_ENFORCE_DETECTION = True

ENABLE_TTA = True
TTA_ROT_DEG = 5
TTA_BRIGHT_FACTOR = 0.10
TTA_MAX_VARIANTS = 5

ENABLE_SMOOTHING = True
CENTERED_SMOOTH_WINDOW = 3

ENABLE_SMOOTHING_SCORE = True
SMOOTH_SCORE_USE_FINAL = True

RUN_DEEPFACE_ALWAYS = False
HSEMOTION_CONFIDENCE_THRESHOLD = 0.65
DEEPFACE_CONFIDENCE_THRESHOLD = 0.70

ENABLE_UNCERTAIN_CLASS = True
UNCERTAIN_MIN_CONF = 0.55


# =============================================================================
# CONFIG HELPERS (YAML)
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
    Override les variables globales depuis config.yaml (si présent).
    On ne touche à rien si les clés n'existent pas.
    """
    global MIN_FACE_SIZE
    global BLUR_MIN_VAR_LAPLACIAN, BRIGHTNESS_MIN, BRIGHTNESS_MAX
    global DEEPFACE_DETECTOR_BACKEND, DEEPFACE_ENFORCE_DETECTION
    global ENABLE_TTA, TTA_ROT_DEG, TTA_BRIGHT_FACTOR, TTA_MAX_VARIANTS
    global ENABLE_SMOOTHING, CENTERED_SMOOTH_WINDOW
    global ENABLE_SMOOTHING_SCORE, SMOOTH_SCORE_USE_FINAL
    global RUN_DEEPFACE_ALWAYS, HSEMOTION_CONFIDENCE_THRESHOLD, DEEPFACE_CONFIDENCE_THRESHOLD
    global ENABLE_UNCERTAIN_CLASS, UNCERTAIN_MIN_CONF

    # Quality
    MIN_FACE_SIZE = int(cfg_get(cfg, "face_detection", "quality_filters", "min_face_size", default=MIN_FACE_SIZE))
    BLUR_MIN_VAR_LAPLACIAN = float(cfg_get(cfg, "emotion_analysis", "quality", "blur_min_laplacian", default=BLUR_MIN_VAR_LAPLACIAN))
    BRIGHTNESS_MIN = float(cfg_get(cfg, "emotion_analysis", "quality", "brightness_min", default=BRIGHTNESS_MIN))
    BRIGHTNESS_MAX = float(cfg_get(cfg, "emotion_analysis", "quality", "brightness_max", default=BRIGHTNESS_MAX))

    # DeepFace settings
    DEEPFACE_DETECTOR_BACKEND = str(cfg_get(cfg, "emotion_analysis", "deepface", "detector_backend", default=DEEPFACE_DETECTOR_BACKEND))
    DEEPFACE_ENFORCE_DETECTION = bool(cfg_get(cfg, "emotion_analysis", "deepface", "enforce_detection", default=DEEPFACE_ENFORCE_DETECTION))

    # TTA
    ENABLE_TTA = bool(cfg_get(cfg, "emotion_analysis", "tta", "enabled", default=ENABLE_TTA))
    TTA_ROT_DEG = float(cfg_get(cfg, "emotion_analysis", "tta", "rotation_deg", default=TTA_ROT_DEG))
    TTA_BRIGHT_FACTOR = float(cfg_get(cfg, "emotion_analysis", "tta", "brightness_factor", default=TTA_BRIGHT_FACTOR))
    TTA_MAX_VARIANTS = int(cfg_get(cfg, "emotion_analysis", "tta", "max_variants", default=TTA_MAX_VARIANTS))

    # Smoothing
    ENABLE_SMOOTHING = bool(cfg_get(cfg, "emotion_analysis", "smoothing", "enabled", default=ENABLE_SMOOTHING))
    CENTERED_SMOOTH_WINDOW = int(cfg_get(cfg, "emotion_analysis", "smoothing", "window", default=CENTERED_SMOOTH_WINDOW))
    ENABLE_SMOOTHING_SCORE = bool(cfg_get(cfg, "emotion_analysis", "smoothing_score", "enabled", default=ENABLE_SMOOTHING_SCORE))
    SMOOTH_SCORE_USE_FINAL = bool(cfg_get(cfg, "emotion_analysis", "smoothing_score", "use_final", default=SMOOTH_SCORE_USE_FINAL))

    # Thresholds / logic
    RUN_DEEPFACE_ALWAYS = bool(cfg_get(cfg, "emotion_analysis", "run_deepface_always", default=RUN_DEEPFACE_ALWAYS))
    HSEMOTION_CONFIDENCE_THRESHOLD = float(cfg_get(cfg, "emotion_analysis", "hsemotion", "confidence_threshold", default=HSEMOTION_CONFIDENCE_THRESHOLD))
    DEEPFACE_CONFIDENCE_THRESHOLD = float(cfg_get(cfg, "emotion_analysis", "deepface", "confidence_threshold", default=DEEPFACE_CONFIDENCE_THRESHOLD))

    ENABLE_UNCERTAIN_CLASS = bool(cfg_get(cfg, "emotion_analysis", "uncertain", "enabled", default=ENABLE_UNCERTAIN_CLASS))
    UNCERTAIN_MIN_CONF = float(cfg_get(cfg, "emotion_analysis", "uncertain", "min_conf", default=UNCERTAIN_MIN_CONF))


# =============================================================================
# UTIL PARSING
# =============================================================================

def parse_frame_index(filename: str) -> int:
    m = re.search(r"frame_(\d+)", filename.lower())
    return int(m.group(1)) if m else -1

def parse_face_id(filename: str) -> int:
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
    return track_id if track_id != -1 else face_id


# =============================================================================
# GLOBAL PERSON ID (UNIQUE ACROSS VIDEOS)
# =============================================================================

def _norm_key(s: str) -> str:
    """Normalise en clé stable: slash/backslash/espaces -> '_'."""
    s = (s or "").replace("\\", "/").strip("/")
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "_")
    return s

def extract_person_folder(rel_dir: str) -> Optional[str]:
    """
    Cherche un segment 'person_0000' dans rel_dir.
    Retourne 'person_0000' si trouvé, sinon None.
    """
    if not rel_dir:
        return None
    parts = rel_dir.replace("\\", "/").split("/")
    for p in parts[::-1]:
        if re.fullmatch(r"person[_-]?\d+", p, flags=re.IGNORECASE):
            num = re.search(r"(\d+)", p).group(1)
            return f"person_{int(num):04d}"
    return None

def extract_video_key(rel_dir: str) -> str:
    """
    Déduit la 'vidéo' à partir de rel_dir en supprimant person_XXXX.
    Exemple:
      bedouk/frames_.../person_0000  -> bedouk_frames_...
    """
    if not rel_dir:
        return "unknown_video"
    parts = [p for p in rel_dir.replace("\\", "/").split("/") if p]
    parts = [p for p in parts if not re.fullmatch(r"person[_-]?\d+", p, flags=re.IGNORECASE)]
    key = "_".join(parts) if parts else "unknown_video"
    return _norm_key(key)

def make_global_person_id(rel_dir: str, identity_id_int: int) -> str:
    """
    Construit un identifiant global unique:
      <video_key>_<person_XXXX>
    """
    video_key = extract_video_key(rel_dir)
    person_folder = extract_person_folder(rel_dir)
    if person_folder is None:
        person_folder = f"person_{max(0, int(identity_id_int)):04d}"
    return f"{video_key}_{person_folder}"


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
    if face_bgr is None or face_bgr.size == 0:
        return {"blur": 0.0, "brightness": 0.0, "contrast": 0.0, "area": 0.0}

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    area = float(face_bgr.shape[0] * face_bgr.shape[1])
    return {"blur": blur, "brightness": brightness, "contrast": contrast, "area": area}

def is_bad_quality(q: Dict[str, float]) -> bool:
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
    variants = [face_bgr]
    if face_bgr is None or face_bgr.size == 0:
        return variants

    variants.append(cv2.flip(face_bgr, 1))

    img = face_bgr.astype(np.float32)
    brighter = np.clip(img * (1.0 + TTA_BRIGHT_FACTOR), 0, 255).astype(np.uint8)
    darker = np.clip(img * (1.0 - TTA_BRIGHT_FACTOR), 0, 255).astype(np.uint8)
    variants.append(brighter)
    variants.append(darker)

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
# SMOOTHING + SCORE
# =============================================================================

def centered_mode(values: List[Optional[str]]) -> Optional[str]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def compute_smoothing_score(
    emotions_window: List[Optional[str]],
    confidences_window: List[float],
) -> Tuple[Optional[str], float, float, float]:
    emo = [e for e in emotions_window if isinstance(e, str) and e.strip()]
    if not emo:
        return None, 0.0, 0.0, 0.0

    counts = Counter(emo)
    smoothed_emotion, top_count = counts.most_common(1)[0]
    total = len(emo)
    vote_ratio = (top_count / total) if total else 0.0

    confs = [max(0.0, min(1.0, _safe_float(c, 0.0))) for c in confidences_window]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0

    smoothing_score = vote_ratio * avg_conf
    return smoothed_emotion, smoothing_score, vote_ratio, avg_conf

def apply_centered_smoothing(entries: List[dict], window: int, key_emotion: str, out_key: str):
    if window % 2 != 1 or window < 3:
        raise ValueError("window doit être impair et >= 3")

    half = window // 2
    n = len(entries)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals = [entries[j].get(key_emotion) for j in range(lo, hi)]
        entries[i][out_key] = centered_mode(vals)

def apply_smoothing_score_on_key(
    entries: List[dict],
    window: int,
    emotion_key: str,
    conf_key: str,
    out_score_key: str,
    out_vote_key: str,
    out_avgconf_key: str,
    out_winsize_key: str,
):
    if window % 2 != 1 or window < 3:
        raise ValueError("window doit être impair et >= 3")

    half = window // 2
    n = len(entries)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        emos = [entries[j].get(emotion_key) for j in range(lo, hi)]
        confs = [entries[j].get(conf_key, 0.0) for j in range(lo, hi)]

        _, score, vote_ratio, avg_conf = compute_smoothing_score(emos, confs)

        entries[i][out_score_key] = float(score)
        entries[i][out_vote_key] = float(vote_ratio)
        entries[i][out_avgconf_key] = float(avg_conf)
        entries[i][out_winsize_key] = int(hi - lo)

def apply_centered_smoothing_per_dir(per_dir_rows: Dict[str, List[dict]], window: int):
    for rel_dir, rows in per_dir_rows.items():
        groups: Dict[int, List[dict]] = {}
        for r in rows:
            iid = int(r.get("identity_id", -1))
            groups.setdefault(iid, []).append(r)

        for _, lst in groups.items():
            lst.sort(key=lambda x: int(x.get("frame_index", -1)))

            apply_centered_smoothing(lst, window=window, key_emotion="hse_emotion", out_key="smoothed_hse_emotion")
            apply_centered_smoothing(lst, window=window, key_emotion="deepface_emotion", out_key="smoothed_deepface_emotion")

            for r in lst:
                sh = r.get("smoothed_hse_emotion")
                sd = r.get("smoothed_deepface_emotion")
                if sh is not None and sd is not None:
                    r["smoothed_final_emotion"] = sh if sh == sd else sh
                else:
                    r["smoothed_final_emotion"] = sh or sd

            if ENABLE_SMOOTHING_SCORE:
                if SMOOTH_SCORE_USE_FINAL:
                    apply_smoothing_score_on_key(
                        lst, window=window,
                        emotion_key="final_emotion",
                        conf_key="final_confidence",
                        out_score_key="smoothing_score",
                        out_vote_key="smoothing_vote_ratio",
                        out_avgconf_key="smoothing_avg_conf_window",
                        out_winsize_key="smoothing_window_size",
                    )
                else:
                    apply_smoothing_score_on_key(
                        lst, window=window,
                        emotion_key="smoothed_final_emotion",
                        conf_key="final_confidence",
                        out_score_key="smoothing_score",
                        out_vote_key="smoothing_vote_ratio",
                        out_avgconf_key="smoothing_avg_conf_window",
                        out_winsize_key="smoothing_window_size",
                    )

                for r in lst:
                    r["was_smoothed_changed"] = (r.get("smoothed_final_emotion") != r.get("final_emotion"))


# =============================================================================
# BACKENDS
# =============================================================================

class DeepFaceEmotionDetector:
    def __init__(self, detector_backend: str = DEEPFACE_DETECTOR_BACKEND, enforce_detection: bool = DEEPFACE_ENFORCE_DETECTION):
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        print(f"[DeepFaceEmotionDetector] Ready (backend={detector_backend}, enforce={enforce_detection})")

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

        emotions, confs = [], []
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


class HSEmotionDetector:
    def __init__(self, device: str = "cpu"):
        self._printed_error = False
        print("[HSEmotionDetector] Chargement du modèle HSEmotion...")
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf", device=device)
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

        emotions, confs = [], []
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
# DECISION LOGIC
# =============================================================================

def decide_final(hse_emotion, hse_conf, df_emotion, df_conf, quality_bad) -> Tuple[Optional[str], float, str, bool]:
    if ENABLE_UNCERTAIN_CLASS:
        low_both = (hse_conf < UNCERTAIN_MIN_CONF) and (df_conf < UNCERTAIN_MIN_CONF)
        disagree = (hse_emotion is not None and df_emotion is not None and hse_emotion != df_emotion)
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
# SKIP HELPERS (MASTER-BASED)
# =============================================================================

def dir_fully_processed(rel_dir: str, img_files: List[str], master_results: dict) -> bool:
    if not img_files:
        return False
    for fn in img_files:
        rel_path = fn if rel_dir == "." else os.path.join(rel_dir, fn)
        if rel_path not in master_results:
            return False
    return True


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
    global_person_id: str
    image_path: str


def analyze_emotions_incremental(faces_root: str, output_root: str, master_json_path: str):
    """
    Pipeline inchangé, paramètres peuvent être overridés via config.yaml.
    """
    os.makedirs(output_root, exist_ok=True)
    master_results = load_master_json(master_json_path)

    hse_detector = HSEmotionDetector(device="cpu")
    df_detector = DeepFaceEmotionDetector(
        detector_backend=DEEPFACE_DETECTOR_BACKEND,
        enforce_detection=DEEPFACE_ENFORCE_DETECTION,
    )

    tasks: List[Task] = []

    for dirpath, _, filenames in os.walk(faces_root):
        rel_dir = os.path.relpath(dirpath, faces_root)

        img_files = [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not img_files:
            continue

        if dir_fully_processed(rel_dir, img_files, master_results):
            print(f"[SKIP] Dossier déjà analysé (master complet): {rel_dir}")
            continue

        for filename in img_files:
            rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)
            if rel_path in master_results:
                continue

            frame_index = parse_frame_index(filename)
            if frame_index == -1:
                continue

            fid = parse_face_id(filename)
            tid = parse_track_id(filename)
            iid = identity_id(fid, tid)
            gid = make_global_person_id(rel_dir, iid)

            image_path = os.path.join(dirpath, filename)

            tasks.append(Task(
                rel_dir=rel_dir,
                rel_path=rel_path,
                filename=filename,
                frame_index=frame_index,
                face_id=fid,
                track_id=tid,
                identity_id=iid,
                global_person_id=gid,
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
                "global_person_id": t.global_person_id,

                "hse_emotion": None,
                "hse_confidence": 0.0,
                "deepface_emotion": None,
                "deepface_confidence": 0.0,
                "agree": False,

                "final_emotion": None,
                "final_confidence": 0.0,
                "final_backend": "too_small",
                "is_uncertain": True,

                "smoothed_hse_emotion": None,
                "smoothed_deepface_emotion": None,
                "smoothed_final_emotion": None,

                "smoothing_score": 0.0,
                "smoothing_vote_ratio": 0.0,
                "smoothing_avg_conf_window": 0.0,
                "smoothing_window_size": 0,
                "was_smoothed_changed": False,

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

        hse_emotion, hse_conf = hse_detector.analyze(image, use_tta=ENABLE_TTA)

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
            "global_person_id": t.global_person_id,

            "hse_emotion": hse_emotion,
            "hse_confidence": float(hse_conf),
            "deepface_emotion": df_emotion,
            "deepface_confidence": float(df_conf),
            "agree": bool(agree),

            "final_emotion": final_emotion,
            "final_confidence": float(final_conf),
            "final_backend": final_backend,
            "is_uncertain": bool(is_uncertain),

            "smoothed_hse_emotion": None,
            "smoothed_deepface_emotion": None,
            "smoothed_final_emotion": None,

            "smoothing_score": 0.0,
            "smoothing_vote_ratio": 0.0,
            "smoothing_avg_conf_window": 0.0,
            "smoothing_window_size": 0,
            "was_smoothed_changed": False,

            "quality_blur": q["blur"],
            "quality_brightness": q["brightness"],
            "quality_contrast": q["contrast"],
            "quality_area": q["area"],
            "bad_quality": bool(bad_q),
        }

        per_dir_rows.setdefault(t.rel_dir, []).append(entry)
        per_dir_json.setdefault(t.rel_dir, {})[t.rel_path] = entry
        master_results[t.rel_path] = entry

    if ENABLE_SMOOTHING:
        apply_centered_smoothing_per_dir(per_dir_rows, window=CENTERED_SMOOTH_WINDOW)

        for rel_dir, rows in per_dir_rows.items():
            for r in rows:
                rp = r["relative_path"]
                if rel_dir in per_dir_json and rp in per_dir_json[rel_dir]:
                    per_dir_json[rel_dir][rp] = r
                master_results[rp] = r

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    fieldnames = [
        "relative_path","filename","frame_index","face_id","track_id","identity_id","global_person_id",
        "hse_emotion","hse_confidence","deepface_emotion","deepface_confidence","agree",
        "final_emotion","final_confidence","final_backend","is_uncertain",
        "smoothed_hse_emotion","smoothed_deepface_emotion","smoothed_final_emotion",
        "smoothing_score","smoothing_vote_ratio","smoothing_avg_conf_window","smoothing_window_size","was_smoothed_changed",
        "quality_blur","quality_brightness","quality_contrast","quality_area","bad_quality",
    ]

    for rel_dir, rows in per_dir_rows.items():
        base_dir = "root" if rel_dir == "." else rel_dir

        run_folder = os.path.join(output_root, base_dir, "latest")
        os.makedirs(run_folder, exist_ok=True)

        csv_path = os.path.join(run_folder, "analyzed_emotions.csv")
        json_path = os.path.join(run_folder, "analyzed_emotions.json")
        final_json_path = os.path.join(run_folder, "analyzed_emotions_final.json")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            csvfile.write(f"# Emotions analysis run at {timestamp}\n")
            csvfile.write(f"# DeepFace backend: {DEEPFACE_DETECTOR_BACKEND}, enforce={DEEPFACE_ENFORCE_DETECTION}\n")
            csvfile.write(f"# TTA: {ENABLE_TTA}, smoothing: {ENABLE_SMOOTHING} (window={CENTERED_SMOOTH_WINDOW})\n")
            csvfile.write(f"# Fallback thresholds: HSE={HSEMOTION_CONFIDENCE_THRESHOLD}, DF={DEEPFACE_CONFIDENCE_THRESHOLD}\n")
            csvfile.write(f"# Smoothing score: {ENABLE_SMOOTHING_SCORE} (use_final={SMOOTH_SCORE_USE_FINAL})\n")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        with open(json_path, mode="w", encoding="utf-8") as jsonfile:
            json.dump(per_dir_json[rel_dir], jsonfile, indent=4, ensure_ascii=False)

        final_only = {}
        for row in rows:
            rp = row["relative_path"]
            final_only[rp] = {
                "final_emotion": row.get("final_emotion"),
                "final_confidence": row.get("final_confidence", 0.0),
                "final_backend": row.get("final_backend"),
                "is_uncertain": row.get("is_uncertain", True),
                "smoothed_final_emotion": row.get("smoothed_final_emotion"),
                "smoothing_score": row.get("smoothing_score", 0.0),
                "smoothing_vote_ratio": row.get("smoothing_vote_ratio", 0.0),
                "smoothing_avg_conf_window": row.get("smoothing_avg_conf_window", 0.0),
                "smoothing_window_size": row.get("smoothing_window_size", 0),
                "was_smoothed_changed": row.get("was_smoothed_changed", False),
                "frame_index": row.get("frame_index"),
                "identity_id": row.get("identity_id"),
                "global_person_id": row.get("global_person_id"),
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


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze emotions from detected faces (VideoEmotion)")
    parser.add_argument("--faces-root", default=None, help="Override faces root.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--master-json", default=None, help="Override master json path.")
    parser.add_argument("--project-root", default=None, help="Racine du projet (défaut: auto).")
    parser.add_argument("--config", default=None, help="Chemin vers config.yaml (défaut: <project-root>/config.yaml).")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    config_path = resolve_from_project(project_root, args.config) if args.config else (project_root / "config.yaml")
    cfg = load_config(config_path)

    # Override globals (config -> globals)
    apply_config_overrides(cfg)

    # Paths par défaut depuis config
    cfg_faces_root = cfg_get(cfg, "paths", "detected_faces", default="data/detected_faces")
    cfg_output_root = cfg_get(cfg, "paths", "emotion_results", default="output/emotion_results")

    faces_root = resolve_from_project(project_root, args.faces_root) if args.faces_root else resolve_from_project(project_root, str(cfg_faces_root))
    output_root = resolve_from_project(project_root, args.output_root) if args.output_root else resolve_from_project(project_root, str(cfg_output_root))

    if args.master_json:
        master_json_path = resolve_from_project(project_root, args.master_json)
    else:
        master_json_path = (output_root / "emotion_results_master.json").resolve()

    if not faces_root.exists():
        print(f"[ERREUR] Dossier faces introuvable: {faces_root}")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    analyze_emotions_incremental(
        faces_root=str(faces_root),
        output_root=str(output_root),
        master_json_path=str(master_json_path),
    )


if __name__ == "__main__":
    main()
