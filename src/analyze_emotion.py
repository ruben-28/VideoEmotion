import os
import cv2
import csv
import json
from datetime import datetime
from fer import FER


def parse_frame_index(filename: str) -> int:
    """
    Extrait l'index de frame depuis un nom du type:
    frame_00012_face_000.jpg -> 12
    """
    try:
        base = filename.split("_face_")[0]  # ex: "frame_00012"
        num_str = base.replace("frame_", "")  # "00012"
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
        json.dump(master, f, indent=4)


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

    # 1) Master global
    master_results = load_master_json(master_json_path)

    # 2) Détecteur d'émotions
    emotion_detector = FER(mtcnn=True)

    # 3) On va regrouper les résultats par sous-dossier de detected_faces
    #    per_dir_rows["frames_XXX"] = [entry1, entry2, ...]
    #    per_dir_json["frames_XXX"] = { rel_path: entry, ... }
    per_dir_rows = {}
    per_dir_json = {}

    # 4) Parcours récursif de faces_root
    for dirpath, _, filenames in os.walk(faces_root):
        rel_dir = os.path.relpath(dirpath, faces_root)  # peut être "." pour la racine

        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # chemin relatif du fichier (incluant éventuellement le sous-dossier)
            if rel_dir == ".":
                rel_path = filename
            else:
                rel_path = os.path.join(rel_dir, filename)

            # Déjà analysé ? -> on saute
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

            emotions = emotion_detector.detect_emotions(image)
            if emotions:
                topEmotion, score = emotion_detector.top_emotion(image)
            else:
                topEmotion, score = None, 0.0

            entry = {
                "relative_path": rel_path,
                "filename": filename,
                "frame_index": frame_index,
                "top_emotion": topEmotion,
                "confidence_score": float(score)
            }

            # Initialiser la structure pour ce dossier si besoin
            if rel_dir not in per_dir_rows:
                per_dir_rows[rel_dir] = []
                per_dir_json[rel_dir] = {}

            per_dir_rows[rel_dir].append(entry)
            per_dir_json[rel_dir][rel_path] = entry

            # Mettre à jour le master
            master_results[rel_path] = entry

    # 5) Si aucun nouveau fichier n'a été analysé
    if not per_dir_rows:
        print("Aucune nouvelle image à analyser. Tout est déjà à jour ✅")
        return

    # 6) On utilise un seul timestamp pour ce run
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # 7) Pour chaque dossier de detected_faces ayant des nouveaux fichiers,
    #    on crée UN sous-dossier avec ce timestamp et on y met emotions.csv / emotions.json
    fieldnames = ["relative_path", "filename", "frame_index", "top_emotion", "confidence_score"]

    for rel_dir, rows in per_dir_rows.items():
        # Construire le dossier de sortie :
        # Si rel_dir == ".", on peut appeler ça "root"
        if rel_dir == ".":
            base_dir = "root"
        else:
            base_dir = rel_dir

        run_folder = os.path.join(output_root, base_dir, timestamp)
        os.makedirs(run_folder, exist_ok=True)

        csv_path = os.path.join(run_folder, "emotions.csv")
        json_path = os.path.join(run_folder, "emotions.json")

        # CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(f"# Emotions analysis run at {timestamp}\n")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        # JSON
        with open(json_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(per_dir_json[rel_dir], jsonfile, indent=4)

        print(f"{len(rows)} images analysées pour le dossier '{rel_dir}' ✅")
        print(f"→ CSV :  {csv_path}")
        print(f"→ JSON : {json_path}")

    # 8) Sauvegarde du master global
    save_master_json(master_results, master_json_path)
    print(f"Master JSON mis à jour : {master_json_path}")


if __name__ == "__main__":
    faces_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces"
    output_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\output\\emotions"
    master_json_path = os.path.join(output_root, "emotions_master.json")

    analyze_emotions_incremental(faces_root, output_root, master_json_path)
