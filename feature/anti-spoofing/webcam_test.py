import cv2
from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    SCRIPT_DIR / "weights" / "best.pt",
    SCRIPT_DIR / "anti_spoofing_yolo_with_aug_final01" / "weights" / "best.pt",
]

model_path = next((path for path in MODEL_PATHS if path.exists()), None)
if model_path is None:
    checked_paths = ", ".join(str(path) for path in MODEL_PATHS)
    raise FileNotFoundError(f"Anti-spoofing model not found. Checked: {checked_paths}")

# Load trained anti-spoofing model
model = YOLO(model_path)
#"C:\Users\leona\runs\classify\anti_spoofing_yolo_light_aug\weights\best.pt" 1st best
# C:\Users\leona\runs\classify\anti_spoofing_yolo_with_aug_final01\weights\best.pt 2nd best
#"C:\Users\leona\runs\classify\anti_spoofing_yolo_light_aug_2\weights\best.pt" 3rd best
#"C:\Users\leona\runs\classify\anti_spoofing_yolo_gpu-13\weights\best.pt" 4th best
#"C:\Users\leona\runs\classify\anti_spoofing_yolo_light_aug_4\weights\best.pt" 5th best (not rlly good)


IMG_SIZE = 224

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read from webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(120, 120)
    )

    for (x, y, w, h) in faces:
        padding = 40

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, frame.shape[1])
        y2 = min(y + h + padding, frame.shape[0])

        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            continue

        results = model.predict(
            source=face_region,
            imgsz=IMG_SIZE,
            verbose=False
        )

        probs = results[0].probs
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = model.names[class_id]

        if class_name == "spoof":
            label = f"SPOOF {confidence:.2f}"
            color = (0, 0, 255)
        else:
            label = f"REAL {confidence:.2f}"
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Anti-Spoofing Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
