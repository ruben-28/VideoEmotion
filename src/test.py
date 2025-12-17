import cv2
from deepface import DeepFace

IMG_PATH = "C:\\Users\\ruben\\Desktop\\VideoEmotion\\data\\detected_faces\\ruben\\frames_04-12-2025_19-04-15\\frame_00005_face_000.jpg"

image = cv2.imread(IMG_PATH)
if image is None:
    print("Impossible de lire l'image :", IMG_PATH)
    exit()

# BGR -> RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

try:
    result = DeepFace.analyze(
        image_rgb,
        actions=['emotion'],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    print("Résultat DeepFace brut :")
    print(result)

    print("\nDominant emotion :", result.get("dominant_emotion"))
    print("Scores :", result.get("emotion"))

except Exception as e:
    print("Erreur DeepFace :", e)