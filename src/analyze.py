import os
import cv2
import csv
import json
from datetime import datetime
import numpy as np
from typing import Tuple, Optional
import traceback

# =====================================================================
# CONFIG
# =====================================================================

# --- TON MODELE (.keras) ---
CUSTOM_MODEL_PATH = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\models\\emotion_faces_v2s_final.keras"

# IMPORTANT: ordre EXACT des classes de ton modèle
CUSTOM_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ✅ Ton modèle attend (224,224,3)
CUSTOM_INPUT_SIZE = (224, 224)
CUSTOM_COLOR_MODE = "rgb"      # ✅ RGB (3 canaux)
CUSTOM_NORMALIZE_0_1 = True    # ✅ souvent correct si entraînement avec rescale=1./255


# =====================================================================
# UTIL
# =====================================================================
def parse_frame_index(filename: str) -> int:
    try:
        base = filename.split("face")[0]
        num_str = base.replace("frame_", "").replace("_", "")
        return int(num_str)
    except Exception:
        return -1


def load_master_json(master_json_path: str) -> dict:
    if os.path.exists(master_json_path):
        with open(master_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_master_json(master: dict, master_json_path: str):
    os.makedirs(os.path.dirname(master_json_path), exist_ok=True)
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=4, ensure_ascii=False)


# =====================================================================
# BACKEND A: HSEMOTION
# =====================================================================
class HSEmotionDetector:
    def __init__(self, device: str = "cpu"):
        self._printed_error = False
        print("[HSEmotionDetector] Chargement du modèle HSEmotion...")
        from hsemotion.facial_emotions import HSEmotionRecognizer

        self.model = HSEmotionRecognizer(
            model_name="enet_b0_8_best_vgaf", device=device
        )
        print("[HSEmotionDetector] Modèle chargé ✅")

    def analyze(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0

        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            emotion, scores = self.model.predict_emotions(img_rgb, logits=False)
            conf = float(np.max(scores)) if scores is not None else 0.0
            return emotion, conf
        except Exception:
            if not self._printed_error:
                self._printed_error = True
                print("[HSEmotionDetector] ERREUR (une seule fois) :")
                traceback.print_exc()
            return None, 0.0


# =====================================================================
# BACKEND B: TON MODELE (.keras)
# =====================================================================
class CustomEmotionDetector:
    """
    Utilise ton modèle entraîné (Keras/TensorFlow) et renvoie (emotion, confidence)
    confidence = proba max (0..1)
    """

    def __init__(
        self,
        model_path: str,
        labels: list,
        input_size: Tuple[int, int],
        color_mode: str = "rgb",
        normalize_0_1: bool = True,
    ):
        print("[CustomEmotionDetector] Chargement du modèle perso...")
        self.labels = labels
        self.input_size = input_size
        self.color_mode = color_mode
        self.normalize_0_1 = normalize_0_1

        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)  # .keras OK

        # ✅ Debug utile : voir exactement ce que le modèle attend
        print("[CustomEmotionDetector] Model input shape:", self.model.input_shape)
        print("[CustomEmotionDetector] Modèle chargé ✅")

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        # Resize vers 224x224
        img = cv2.resize(img_bgr, self.input_size, interpolation=cv2.INTER_AREA)

        if self.color_mode == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32)
            if self.normalize_0_1:
                img /= 255.0
            img = np.expand_dims(img, axis=-1)  # (H,W,1)

        elif self.color_mode == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            if self.normalize_0_1:
                img /= 255.0
            # (H,W,3)

        else:
            raise ValueError("CUSTOM_COLOR_MODE doit être 'gray' ou 'rgb'.")

        # Batch dimension -> (1,H,W,C)
        img = np.expand_dims(img, axis=0)
        return img

    def analyze(self, img_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0

        try:
            x = self._preprocess(img_bgr)
            preds = self.model.predict(x, verbose=0)

            if isinstance(preds, list):
                preds = preds[0]

            preds = np.array(preds).squeeze()  # (N,)

            if preds.ndim != 1 or preds.size != len(self.labels):
                print(
                    f"[CustomEmotionDetector] Sortie inattendue: shape={preds.shape}, "
                    f"labels={len(self.labels)}"
                )
                return None, 0.0

            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            emotion = self.labels[idx]
            return emotion, conf

        except Exception as e:
            print(f"[CustomEmotionDetector] Erreur analyse modèle perso: {e}")
            return None, 0.0


# =====================================================================
# COMPARATIF
# =====================================================================
def compare_predictions(hse_emotion, hse_conf, custom_emotion, custom_conf):
    agree = (
        hse_emotion is not None
        and custom_emotion is not None
        and hse_emotion == custom_emotion
    )
    conf_gap = float(abs(hse_conf - custom_conf))

    if custom_conf > hse_conf:
        winner = "custom"
    elif hse_conf > custom_conf:
        winner = "hsemotion"
    else:
        winner = "tie"

    return agree, conf_gap, winner


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def analyze_emotions_incremental_compare(
    faces_root: str, output_root: str, master_json_path: str
):
    os.makedirs(output_root, exist_ok=True)
    master_results = load_master_json(master_json_path)

    hse_detector = HSEmotionDetector(device="cpu")
    custom_detector = CustomEmotionDetector(
        model_path=CUSTOM_MODEL_PATH,
        labels=CUSTOM_LABELS,
        input_size=CUSTOM_INPUT_SIZE,
        color_mode=CUSTOM_COLOR_MODE,
        normalize_0_1=CUSTOM_NORMALIZE_0_1,
    )

    per_dir_rows = {}
    per_dir_json = {}

    for dirpath, _, filenames in os.walk(faces_root):
        rel_dir = os.path.relpath(dirpath, faces_root)

        for filename in filenames:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)

            # Déjà analysé ? skip (incrémental)
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

            # Toujours les 2 modèles
            hse_emotion, hse_conf = hse_detector.analyze(image)
            custom_emotion, custom_conf = custom_detector.analyze(image)

            agree, conf_gap, winner = compare_predictions(
                hse_emotion, hse_conf, custom_emotion, custom_conf
            )

            entry = {
                "relative_path": rel_path,
                "filename": filename,
                "frame_index": frame_index,

                "hse_emotion": hse_emotion,
                "hse_confidence": float(hse_conf),

                "custom_emotion": custom_emotion,
                "custom_confidence": float(custom_conf),

                "agree": bool(agree),
                "confidence_gap": float(conf_gap),
                "winner_backend": winner,
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

    fieldnames = [
        "relative_path",
        "filename",
        "frame_index",
        "hse_emotion",
        "hse_confidence",
        "custom_emotion",
        "custom_confidence",
        "agree",
        "confidence_gap",
        "winner_backend",
    ]

    for rel_dir, rows in per_dir_rows.items():
        base_dir = "root" if rel_dir == "." else rel_dir
        run_folder = os.path.join(output_root, base_dir, timestamp)
        os.makedirs(run_folder, exist_ok=True)

        csv_path = os.path.join(run_folder, "emotions_compare.csv")
        json_path = os.path.join(run_folder, "emotions_compare.json")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            csvfile.write(f"# Emotions compare run at {timestamp}\n")
            csvfile.write(f"# Custom model: {CUSTOM_MODEL_PATH}\n")
            csvfile.write(f"# Custom labels: {CUSTOM_LABELS}\n")
            csvfile.write(
                f"# Custom input_size: {CUSTOM_INPUT_SIZE}, mode={CUSTOM_COLOR_MODE}, norm0_1={CUSTOM_NORMALIZE_0_1}\n"
            )
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        with open(json_path, mode="w", encoding="utf-8") as jsonfile:
            json.dump(per_dir_json[rel_dir], jsonfile, indent=4, ensure_ascii=False)

        print(f"{len(rows)} faces analysées pour le dossier '{rel_dir}' ✅")
        print(f"→ CSV :  {csv_path}")
        print(f"→ JSON : {json_path}")

    save_master_json(master_results, master_json_path)
    print(f"Master JSON mis à jour : {master_json_path}")


if __name__ == "__main__":
    faces_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces"
    output_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\output\\emotions"
    master_json_path = os.path.join(output_root, "emotions_master_compare.json")

    analyze_emotions_incremental_compare(faces_root, output_root, master_json_path)
