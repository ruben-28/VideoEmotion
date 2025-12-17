import os
import cv2
import csv
import json
from datetime import datetime
import numpy as np
from typing import List, Tuple, Optional
import traceback


from deepface import DeepFace


# =====================================================================
# CONFIG
# =====================================================================
USE_DEEPFACE = False        # DeepFace backend
USE_HSEMOTION = True      # HSEmotion backend (mets True pour l'activer)

DEEPFACECONFIDENCE_THRESHOLD = 0.7  # Seuil DeepFace (0..100 en général)

# =====================================================================
# UTIL
# =====================================================================
def parse_frame_index(filename: str) -> int:
    """
    Extrait l'index de frame depuis un nom du type:
    frame_00012_face_000.jpg -> 12
    (chez toi: frame_00012face000.jpg marche aussi)
    """
    try:
        base = filename.split("face")[0]  # ex: "frame_00012"
        num_str = base.replace("frame_", "").replace("_", "")
        return int(num_str)
    except Exception:
        return -1


def load_master_json(master_json_path: str) -> dict:
    """Charge le JSON maître (tous les fichiers déjà analysés)."""
    if os.path.exists(master_json_path):
        with open(master_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_master_json(master: dict, master_json_path: str):
    """Sauvegarde le JSON maître mis à jour."""
    os.makedirs(os.path.dirname(master_json_path), exist_ok=True)
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=4, ensure_ascii=False)


# =====================================================================
# BACKEND 1: DEEPFACE
# =====================================================================
class DeepFaceEmotionDetector:
    """
    Utilise DeepFace pour prédire l'émotion dominante d'un visage recadré.
    """

    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        print("[DeepFaceEmotionDetector] Modèle DeepFace prêt pour l'analyse d'émotions.")

    def analyze(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Retourne (emotion_dominante, confiance).
        confidence: en général DeepFace renvoie des scores 0..100 pour les émotions.
        """
        if img is None or img.size == 0:
            return None, 0.0

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = DeepFace.analyze(
                img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )

            # dict ou list[dict]
            res0 = pred[0] if isinstance(pred, list) and pred else pred
            if not isinstance(res0, dict):
                return None, 0.0

            dominant_emotion = res0.get('dominant_emotion', None)
            emotion_scores = res0.get('emotion', {}) or {}

            score = float(emotion_scores.get(dominant_emotion, 0.0)) if dominant_emotion else 0.0
            return dominant_emotion, score

        except Exception as e:
            print(f"[DeepFaceEmotionDetector] Erreur lors de l'analyse DeepFace: {e}")
            return None, 0.0


# =====================================================================
# BACKEND 2: HSEMOTION
# =====================================================================
class HSEmotionDetector:
    """
    Wrapper HSEmotion.
    Retourne (emotion_dominante, confiance).
    """
    def __init__(self, device: str = "cpu"):
        self._printed_error = False
        print("[HSEmotionDetector] Chargement du modèle HSEmotion...")
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.model = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
        print("[HSEmotionDetector] Modèle chargé ✅")

    def analyze(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        if img is None or img.size == 0:
            return None, 0.0
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emotion, scores = self.model.predict_emotions(img_rgb, logits=False)

            conf = float(np.max(scores)) if scores is not None else 0.0
            return emotion, conf

        except Exception:
            if not self._printed_error:
                self._printed_error = True
                print("[HSEmotionDetector] ERREUR RÉELLE (une seule fois) :")
                traceback.print_exc()
            return None, 0.0

# =====================================================================
# SELECTOR
# =====================================================================
def build_emotion_detector():
    if USE_HSEMOTION:
        return HSEmotionDetector(device="cpu")
    if USE_DEEPFACE:
        return DeepFaceEmotionDetector()
    # fallback
    return DeepFaceEmotionDetector()


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def analyze_emotions_incremental(faces_root: str,
                                 output_root: str,
                                 master_json_path: str):
    """
    Analyse incrémentale de toutes les images de faces_root.

    - Ne traite que les nouvelles images (pas dans le JSON maître).
    - Pour CHAQUE sous-dossier de faces_root, crée (si nouvelles images) :
        output_root/<rel_dir>/<timestamp>/emotions.csv
        output_root/<rel_dir>/<timestamp>/emotions.json
    - Met à jour un JSON maître global.
    """

    os.makedirs(output_root, exist_ok=True)

    master_results = load_master_json(master_json_path)

    emotion_detector = build_emotion_detector()

    per_dir_rows = {}
    per_dir_json = {}

    for dirpath, _, filenames in os.walk(faces_root):
        rel_dir = os.path.relpath(dirpath, faces_root)

        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)

            if rel_path in master_results:
                continue

            frame_index = parse_frame_index(filename)
            if frame_index == -1:
                continue

            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erreur: impossible de lire {image_path}")
                continue

            topEmotion, score = emotion_detector.analyze(image)

            # (Optionnel) appliquer un seuil pour marquer "uncertain"
            # DeepFace score souvent en 0..100. HSEmotion peut être 0..1.
            # On normalise grossièrement: si score > 1.5 on suppose 0..100
            score_norm = score / 100.0 if score > 1.5 else score

            if topEmotion is None:
                topEmotion = None
                score_norm = 0.0

            # si DeepFace et seuil activé
            if USE_DEEPFACE and not USE_HSEMOTION:
                if score < (DEEPFACECONFIDENCE_THRESHOLD * 100.0):  # seuil 0.7 -> 70
                    # tu peux choisir de mettre "uncertain" au lieu de None
                    pass

            entry = {
                "relative_path": rel_path,
                "filename": filename,
                "frame_index": frame_index,
                "top_emotion": topEmotion,
                "confidence_score": float(score_norm),
                "backend": "hsemotion" if USE_HSEMOTION else "deepface"
            }

            if rel_dir not in per_dir_rows:
                per_dir_rows[rel_dir] = []
                per_dir_json[rel_dir] = {}

            per_dir_rows[rel_dir].append(entry)
            per_dir_json[rel_dir][rel_path] = entry
            master_results[rel_path] = entry

    if not per_dir_rows:
        print("Aucune nouvelle image à analyser. Tout est déjà à jour ✅")
        return

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    fieldnames = ["relative_path", "filename", "frame_index", "top_emotion", "confidence_score", "backend"]

    for rel_dir, rows in per_dir_rows.items():
        base_dir = "root" if rel_dir == "." else rel_dir

        run_folder = os.path.join(output_root, base_dir, timestamp)
        os.makedirs(run_folder, exist_ok=True)

        csv_path = os.path.join(run_folder, "emotions.csv")
        json_path = os.path.join(run_folder, "emotions.json")

        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(f"# Emotions analysis run at {timestamp}\n")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        with open(json_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(per_dir_json[rel_dir], jsonfile, indent=4, ensure_ascii=False)

        print(f"{len(rows)} images analysées pour le dossier '{rel_dir}' ✅")
        print(f"→ CSV :  {csv_path}")
        print(f"→ JSON : {json_path}")

    save_master_json(master_results, master_json_path)
    print(f"Master JSON mis à jour : {master_json_path}")


if __name__ == "__main__":
    # test_path = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces\\ytb2\\frames_17-12-2025_12-37-14\\frame_00001_face_000.jpg"
    # img = cv2.imread(test_path)
    # det = HSEmotionDetector(device="cpu")
    # print("TEST:", det.analyze(img))
    # exit()

    faces_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces"
    output_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\output\\emotions"
    master_json_path = os.path.join(output_root, "emotions_master.json")

    analyze_emotions_incremental(faces_root, output_root, master_json_path)