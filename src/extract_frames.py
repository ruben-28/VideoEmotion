import cv2
import os
from datetime import datetime

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video into a timestamped folder inside output_folder.
    """

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_folder = os.path.join(output_folder, f"frames_{timestamp}")

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps / frame_rate), 1)
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames to {output_folder}")
    return True


def extract_all_new_videos(videos_dir, extracted_root, frame_rate=1):
    """
    Parcourt tous les fichiers vidéo dans videos_dir.
    Pour chaque vidéo non encore traitée, crée un dossier et extrait les frames.
    """

    os.makedirs(extracted_root, exist_ok=True)

    for filename in os.listdir(videos_dir):
        if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        video_path = os.path.join(videos_dir, filename)
        video_name = os.path.splitext(filename)[0]  # ex: "WIN_20251204_15_39_22_Pro"

        # Dossier prévu pour cette vidéo
        video_output_dir = os.path.join(extracted_root, video_name)

        # Vérifier si cette vidéo a déjà été traitée
        if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
            print(f"[SKIP] Les frames de '{filename}' existent déjà.")
            continue

        # Créer dossier de la vidéo
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"[PROCESS] Extraction des frames de '{filename}'...")
        extract_frames(video_path, video_output_dir, frame_rate=frame_rate)


if __name__ == "__main__":
    videos_dir = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\videos"
    extracted_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\extracted_frames"

    extract_all_new_videos(videos_dir, extracted_root, frame_rate=1)
