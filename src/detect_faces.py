import os
import cv2

def is_overlapping(boxA, boxB, overlap_threshold=0.3):
    """
    Check if two bounding boxes overlap based on Intersection over Union (IoU).
    boxA, boxB = (x, y, w, h)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    if (boxAArea + boxBArea - interArea) == 0:
        return False

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou > overlap_threshold


def remove_overlapping_boxes(boxes, overlap_threshold=0.3):
    """
    Remove overlapping bounding boxes based on IoU.
    boxes = [(x, y, w, h), ...]
    """
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])  # sort by x
    non_overlapping_boxes = []

    while boxes:
        current_box = boxes.pop(0)
        non_overlapping_boxes.append(current_box)
        boxes = [box for box in boxes if not is_overlapping(current_box, box, overlap_threshold)]

    return non_overlapping_boxes


def detect_faces_in_all_frames(
    extracted_frames_root,
    detected_faces_root,
    face_cascade_path=r"C:\Users\ruben\Desktop\VideoEmotion\models\haarcascade_frontalface_default.xml",
    profile_cascade_path=r"C:\Users\ruben\Desktop\VideoEmotion\models\haarcascade_profileface.xml",
    overlap_threshold=0.3,
):
    """
    Parcourt tous les fichiers images dans extracted_frames_root,
    détecte les visages uniquement pour les images PAS encore traitées,
    et sauvegarde les faces dans detected_faces_root en gardant la même structure.

    - extracted_frames_root : dossier racine des frames extraites
    - detected_faces_root   : dossier racine où stocker les visages
    """

    # Charger les cascades une seule fois
    frontal_cascade = cv2.CascadeClassifier(face_cascade_path)
    profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

    if frontal_cascade.empty():
        print(f"Error: Could not load face cascade from {face_cascade_path}")
        return
    
    if profile_cascade.empty():
        print(f"Error: Could not load profile cascade from {profile_cascade_path}")
        return

    # Parcourt récursivement extracted_frames_root
    for dirpath, dirnames, filenames in os.walk(extracted_frames_root):
        # Chemin relatif par rapport à la racine
        rel_path = os.path.relpath(dirpath, extracted_frames_root)

        # Dossier de sortie correspondant (même structure)
        output_folder = os.path.join(detected_faces_root, rel_path)
        os.makedirs(output_folder, exist_ok=True)

        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            base_name, _ = os.path.splitext(filename)

            # Vérifier si cette image a déjà été traitée
            already_done = False
            if os.path.exists(output_folder):
                for f in os.listdir(output_folder):
                    if f.startswith(base_name + "_face_"):
                        already_done = True
                        break

            if already_done:
                # Tu peux enlever ce print si ça spam trop
                print(f"[SKIP] {os.path.join(rel_path, filename)} déjà traitée.")
                continue

            # Charger l'image
            img_path = os.path.join(dirpath, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not read image {img_path}")
                continue

            # Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h_img, w_img = gray.shape[0:2]

            # Détection frontal
            faces_frontal = frontal_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Profil gauche
            faces_profile = profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Profil droit en flip
            gray_flipped = cv2.flip(gray, 1)
            faces_profile_flipped = profile_cascade.detectMultiScale(
                gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Corriger les coordonnées du flip
            faces_profile_from_flip = []
            for (x, y, w, h) in faces_profile_flipped:
                x_corrected = w_img - x - w
                faces_profile_from_flip.append((x_corrected, y, w, h))

            # Fusion de toutes les boxes
            faces = list(faces_frontal) + list(faces_profile) + list(faces_profile_from_flip)
            faces = remove_overlapping_boxes(faces, overlap_threshold=overlap_threshold)

            if len(faces) == 0:
                print(f"No faces detected in {os.path.join(rel_path, filename)}")
                continue

            # Sauvegarder chaque visage
            for i, (x, y, w, h) in enumerate(faces):
                face_img = image[y:y+h, x:x+w]
                face_filename = os.path.join(
                    output_folder,
                    f"{base_name}_face_{i:03d}.jpg"
                )
                cv2.imwrite(face_filename, face_img)

            print(f"[OK] {len(faces)} faces saved for {os.path.join(rel_path, filename)} in {output_folder}")


if __name__ == "__main__":
    extracted_frames_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\extracted_frames"
    detected_faces_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces"

    detect_faces_in_all_frames(extracted_frames_root, detected_faces_root)
