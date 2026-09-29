import cv2
import numpy as np
import time
import os

YUNET_MODEL = "yunet.onnx"
SFACE_MODEL = "sface.onnx"
REFERENCE_IMAGE = "sammy1.jpeg"

MATCH_THRESHOLD = 0.36

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (640, 480),
    0.3,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def load_reference():
    image = cv2.imread(REFERENCE_IMAGE)

    if image is None:
        raise RuntimeError("Could not load reference image")

    h, w = image.shape[:2]
    scale = min(640 / w, 640 / h)

    image = cv2.resize(
        image,
        (int(w * scale), int(h * scale))
    )

    h, w = image.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    if faces is None:
        raise RuntimeError("No face found in reference image")

    face = max(faces, key=lambda f: f[-1])

    aligned = recognizer.alignCrop(image, face)

    return recognizer.feature(aligned)


def open_camera():
    while True:
        print("Trying to open camera...")

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            time.sleep(1)

            ret, frame = cap.read()

            if ret:
                print("Camera connected")
                return cap

            cap.release()

        print("Camera unavailable. Retrying in 2 seconds...")
        time.sleep(2)


reference_embedding = load_reference()

print("Reference embedding loaded")

cap = open_camera()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera disconnected")

        cap.release()

        while not os.path.exists("/dev/video0"):
            print("Waiting for /dev/video0...")
            time.sleep(2)

        cap = open_camera()
        continue

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, fw, fh = face[:4]

            aligned = recognizer.alignCrop(frame, face)

            embedding = recognizer.feature(aligned)

            similarity = cosine_similarity(
                reference_embedding,
                embedding
            )

            if similarity >= MATCH_THRESHOLD:
                label = f"SAMMY {similarity:.2f}"
            else:
                label = f"UNKNOWN {similarity:.2f}"

            x = int(x)
            y = int(y)
            fw = int(fw)
            fh = int(fh)

            cv2.rectangle(
                frame,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Live Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
