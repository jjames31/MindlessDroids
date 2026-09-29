import cv2

YUNET_MODEL = "yunet.onnx"
SFACE_MODEL = "sface.onnx"

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.3,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

def get_embedding(path):
    image = cv2.imread(path)

    if image is None:
        raise RuntimeError(f"Could not load {path}")

    h, w = image.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        raise RuntimeError(f"No face detected in {path}")

    # Use the highest-confidence face
    face = max(faces, key=lambda f: f[-1])

    aligned = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned)

    return embedding

sammy1 = get_embedding("sammy1.jpeg")
sammy2 = get_embedding("sammy2.jpeg")
other = get_embedding("test_face.jpg")

same_score = recognizer.match(
    sammy1,
    sammy2,
    cv2.FaceRecognizerSF_FR_COSINE
)

different_score = recognizer.match(
    sammy1,
    other,
    cv2.FaceRecognizerSF_FR_COSINE
)

print(f"Sammy 1 vs Sammy 2: {same_score:.4f}")
print(f"Sammy 1 vs Other:   {different_score:.4f}")
