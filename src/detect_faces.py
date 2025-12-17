import os
import cv2
import mediapipe as mp

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
def detect_faces_mediapipe(image, face_detection, overlap_threshold=0.3):
    """
    Utilise MediaPipe pour détecter les visages dans une image BGR (OpenCV).
    Retourne une liste de bounding boxes en pixels : [(x, y, w, h), ...]
    """
    h_img, w_img = image.shape[:2]

    # MediaPipe travaille en RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    boxes = []
    if results.detections:
        for det in results.detections:
            rel_box = det.location_data.relative_bounding_box
            # Coordonnées relatives (0–1) -> pixels
            x_min = int(rel_box.xmin * w_img)
            y_min = int(rel_box.ymin * h_img)
            w = int(rel_box.width * w_img)
            h = int(rel_box.height * h_img)

            # On s’assure que la box reste dans l’image
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            if x_min + w > w_img:
                w = w_img - x_min
            if y_min + h > h_img:
                h = h_img - y_min

            if w <= 0 or h <= 0:
                continue

            boxes.append((x_min, y_min, w, h))

    # Filtrer les overlaps comme avant
    boxes = remove_overlapping_boxes(boxes, overlap_threshold=overlap_threshold)
    return boxes

def _clip(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def crop_with_margin(image, x, y, w, h, margin_ratio=0.25):
    """
    Retourne (crop, (x0, y0)) où (x0,y0) est l'offset du crop dans l'image originale.
    margin_ratio=0.25 => +25% de marge autour de la bbox.
    """
    H, W = image.shape[:2]
    m = int(max(w, h) * margin_ratio)

    x0 = _clip(x - m, 0, W - 1)
    y0 = _clip(y - m, 0, H - 1)
    x1 = _clip(x + w + m, 0, W)
    y1 = _clip(y + h + m, 0, H)

    if x1 <= x0 or y1 <= y0:
        return None, (0, 0)

    return image[y0:y1, x0:x1], (x0, y0)

def refine_bbox_with_facemesh(image, bbox, face_mesh, outer_margin=0.25, inner_margin=0.10):
    """
    Tente d'améliorer bbox=(x,y,w,h) avec FaceMesh.
    - outer_margin : marge pour créer le crop large (où FaceMesh travaille)
    - inner_margin : marge ajoutée au rectangle issu des landmarks
    Retourne une bbox raffinée (x,y,w,h) ou None si FaceMesh échoue.
    """
    x, y, w, h = bbox
    H, W = image.shape[:2]

    crop_large, (ox, oy) = crop_with_margin(image, x, y, w, h, margin_ratio=outer_margin)
    if crop_large is None:
        return None

    # FaceMesh attend du RGB
    crop_rgb = cv2.cvtColor(crop_large, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(crop_rgb)

    if not res.multi_face_landmarks:
        return None

    # On prend le premier visage (max_num_faces=1 de préférence)
    lm = res.multi_face_landmarks[0].landmark
    ch, cw = crop_large.shape[:2]

    xs = [p.x * cw for p in lm]
    ys = [p.y * ch for p in lm]

    x_min = int(min(xs))
    x_max = int(max(xs))
    y_min = int(min(ys))
    y_max = int(max(ys))

    # Ajoute une marge autour des landmarks
    mw = int((x_max - x_min) * inner_margin)
    mh = int((y_max - y_min) * inner_margin)

    x_min = _clip(x_min - mw, 0, cw - 1)
    y_min = _clip(y_min - mh, 0, ch - 1)
    x_max = _clip(x_max + mw, 0, cw)
    y_max = _clip(y_max + mh, 0, ch)

    rw = x_max - x_min
    rh = y_max - y_min
    if rw <= 0 or rh <= 0:
        return None

    # Convertit en coords image originale
    rx = ox + x_min
    ry = oy + y_min

    # Clip final
    rx = _clip(rx, 0, W - 1)
    ry = _clip(ry, 0, H - 1)
    rw = _clip(rw, 1, W - rx)
    rh = _clip(rh, 1, H - ry)

    return (rx, ry, rw, rh)


def detect_faces_in_all_frames(
    extracted_frames_root,
    detected_faces_root,
    overlap_threshold=0.3,
    model_selection=1,
    min_detection_confidence=0.5,
):
    """
    Parcourt tous les fichiers images dans extracted_frames_root,
    détecte les visages (MediaPipe) uniquement pour les images PAS encore traitées,
    et sauvegarde les faces dans detected_faces_root en gardant la même structure.

    - extracted_frames_root : dossier racine des frames extraites
    - detected_faces_root   : dossier racine où stocker les visages
    """

    # mp_face = mp.solutions.face_detection

    # # On crée une fois l’objet FaceDetection et on le réutilise
    # with mp_face.FaceDetection(
    #     model_selection=model_selection,
    #     min_detection_confidence=min_detection_confidence,
    # ) as face_detection:
    mp_face = mp.solutions.face_detection
    mp_mesh = mp.solutions.face_mesh

    with mp_face.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    ) as face_detection, mp_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        # Parcourt récursivement extracted_frames_root
        for dirpath, dirnames, filenames in os.walk(extracted_frames_root):
            # Chemin relatif par rapport à la racine
            rel_path = os.path.relpath(dirpath, extracted_frames_root)

            # Dossier de sortie correspondant (même structure)
            output_folder = os.path.join(detected_faces_root, rel_path)
            os.makedirs(output_folder, exist_ok=True)

            for filename in filenames:
                if not filename.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
                ):
                    continue

                base_name, _ = os.path.splitext(filename)

                # Vérifier si cette image a déjà été traitée
                already_done = False
                if os.path.exists(output_folder):
                    for f in os.listdir(output_folder):
                        if f.startswith(base_name + "face"):
                            already_done = True
                            break

                if already_done:
                    print(f"[SKIP] {os.path.join(rel_path, filename)} déjà traitée.")
                    continue

                # Charger l'image
                img_path = os.path.join(dirpath, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error: Could not read image {img_path}")
                    continue

                # Détection des visages avec MediaPipe
                faces = detect_faces_mediapipe(
                    image,
                    face_detection=face_detection,
                    overlap_threshold=overlap_threshold,
                )

                if len(faces) == 0:
                    print(f"No faces detected in {os.path.join(rel_path, filename)}")
                    continue

                # Sauvegarder chaque visage comme avant
                # for i, (x, y, w, h) in enumerate(faces):
                #     face_img = image[y:y + h, x:x + w]
                #     face_filename = os.path.join(
                #         output_folder,
                #         f"{base_name}face{i:03d}.jpg"
                #     )
                #     cv2.imwrite(face_filename, face_img)
                
                for i, (x, y, w, h) in enumerate(faces):
                    refined = refine_bbox_with_facemesh(image, (x, y, w, h), face_mesh)

                    # FaceMesh OK => bbox raffinée, sinon fallback bbox d'origine
                    if refined is not None:
                        x2, y2, w2, h2 = refined
                    else:
                        x2, y2, w2, h2 = (x, y, w, h)

                    face_img = image[y2:y2 + h2, x2:x2 + w2]
                    face_filename = os.path.join(output_folder, f"{base_name}face{i:03d}.jpg")
                    cv2.imwrite(face_filename, face_img)


                print(
                    f"[OK] {len(faces)} faces saved for "
                    f"{os.path.join(rel_path, filename)} in {output_folder}"
                )

if __name__ == "__main__":
    extracted_frames_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\extracted_frames"
    detected_faces_root = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces"

    detect_faces_in_all_frames(extracted_frames_root, detected_faces_root)
