import numpy as np
import os
import cv2 as cv
from pathlib import Path
from datetime import datetime
import csv
import shutil
import tempfile
import importlib.util
from tensorflow import keras
from queue import Queue, Empty
from threading import Thread, Event
from ultralytics import YOLO

# Configuration
DB_PATH = "faces_db"
BASE_DIR = Path(__file__).resolve().parent
BACKBONE_PATH = BASE_DIR / "feature" / "facial-recognition" / "cnnbackboneV8.py"
MODEL_PATHS = [
    BASE_DIR / "feature" / "facial-recognition" / "face_embedding_modelV8.keras",
    BASE_DIR / "face_embedding_modelV8.keras",
    Path("feature/facial-recognition/face_embedding_modelV8.keras")
]
EMOTION_MODEL_PATHS = [
    BASE_DIR / "feature" / "emotion-detection" / "emotion_model.keras",
    BASE_DIR / "emotion_model.keras",
    Path("feature/emotion-detection/emotion_model.keras")
]
ANTI_SPOOF_MODEL_PATHS = [
    BASE_DIR / "feature" / "anti-spoofing" / "weights" / "best.pt",
    BASE_DIR / "feature" / "anti-spoofing" / "anti_spoofing_yolo_with_aug_final01" / "weights" / "best.pt",
    BASE_DIR / "trained_model" / "anti_spoofing_yolo_with_aug_final01" / "weights" / "best.pt",
    Path("feature/anti-spoofing/weights/best.pt")
]
EMOTION_CLASS_NAMES = ["angry", "happy", "neutral", "sad"]
THRESHOLD = 1.3
PROCESS_FRAMES = 5
IMG_SIZE = (64, 64)
EMOTION_IMG_SIZE = (48, 48)
ANTI_SPOOF_IMG_SIZE = 224
MIN_FACE_SIZE = (40, 40)

# Look for the face embedding model in multiple locations to accommodate different setups
def resolve_model_path():
    for model_path in MODEL_PATHS:
        if model_path.exists():
            return model_path
    paths = ", ".join(str(path) for path in MODEL_PATHS)
    raise FileNotFoundError(f"Could not find face embedding model. Checked: {paths}")

# Look for the emotion model in multiple locations to accommodate different setups
def resolve_emotion_model_path():
    for model_path in EMOTION_MODEL_PATHS:
        if model_path.exists():
            return model_path
    paths = ", ".join(str(path) for path in EMOTION_MODEL_PATHS)
    raise FileNotFoundError(f"Could not find emotion model. Checked: {paths}")

# Look for the anti-spoofing model in multiple locations to accommodate different setups
def resolve_anti_spoof_model_path():
    for model_path in ANTI_SPOOF_MODEL_PATHS:
        if model_path.exists():
            return model_path
    paths = ", ".join(str(path) for path in ANTI_SPOOF_MODEL_PATHS)
    raise FileNotFoundError(f"Could not find anti-spoofing model. Checked: {paths}")

# Load the YOLO-based anti-spoofing model using ultralytics library
def load_anti_spoof_model(model_path):
    return YOLO(model_path)

# Check if a file is in HDF5 format by reading its header
def is_hdf5_file(filepath):
    with open(filepath, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"

# Dynamically import the backbone model from cnnbackboneV8.py and create the embedding model
def create_backbone_model():
    spec = importlib.util.spec_from_file_location("cnnbackboneV8", BACKBONE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import backbone model from {BACKBONE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "create_embedding_model"):
        return module.create_embedding_model()
    if hasattr(module, "model"):
        return module.model

    raise AttributeError(
        f"{BACKBONE_PATH} must define either create_embedding_model() or a model variable."
    )


# Load the face embedding model, handling both Keras 3 format and legacy HDF5 format saved with a .keras extension
def load_embedding_model(model_path):
    load_path = model_path

    # This model is named .keras but is stored in legacy HDF5 format.
    # Keras 3 chooses the loader from the extension, so give it a .h5 path.
    if model_path.suffix.lower() == ".keras" and is_hdf5_file(model_path):
        temp_path = Path(tempfile.gettempdir()) / f"{model_path.stem}.h5"
        if not temp_path.exists() or temp_path.stat().st_mtime < model_path.stat().st_mtime:
            shutil.copy2(model_path, temp_path)
        load_path = temp_path
        print(f"Detected HDF5 model saved with .keras extension; loading via {load_path}")

    try:
        return keras.models.load_model(str(load_path), compile=False, safe_mode=False)
    except ValueError as e:
        if "bad marshal data" not in str(e):
            raise

        print("Could not deserialize the saved Lambda layer; loading architecture from cnnbackboneV8.py.")
        model = create_backbone_model()
        model.load_weights(str(load_path))
        return model

# Calculate squared Euclidean distance, matching feature/facial-recognition/inference.py
#def embedding_distance(a, b):
#    a = np.array(a, dtype=np.float32).reshape(-1)
#    b = np.array(b, dtype=np.float32).reshape(-1)
#    return float(np.sum(np.square(a - b)))

# Calculate cosine distance, matching feature/facial-recognition/inference.py
def embedding_distance(a, b):
    a = np.array(a, dtype=np.float32).reshape(-1)
    b = np.array(b, dtype=np.float32).reshape(-1)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float('inf')
    return 1 - (dot_product / (norm_a * norm_b))

# Preprocess the face image for embedding extraction: read, convert to RGB, resize, normalize, and add batch dimension
def preprocess_face(img):
    if isinstance(img, (str, os.PathLike)):
        img = cv.imread(str(img))
        if img is None:
            return None

    if img.size == 0:
        return None

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    resized = cv.resize(rgb, IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

# Extract the face embedding from the image using the loaded model
def embed(img, model):
    face = preprocess_face(img)
    if face is None:
        return None

    return model.predict(face, verbose=0)[0]


# Preprocess the face image for emotion detection: convert to grayscale, resize to 48x48, and add batch and channel dimensions
def preprocess_emotion_face(face_img):
    if face_img.size == 0:
        return None

    gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, EMOTION_IMG_SIZE)
    return resized.reshape(1, EMOTION_IMG_SIZE[0], EMOTION_IMG_SIZE[1], 1).astype(np.float32)

def predict_emotion(face_img, emotion_model):
    face = preprocess_emotion_face(face_img)
    if face is None:
        return "?", 0.0

    probs = emotion_model.predict(face, verbose=0)[0]
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    if class_id >= len(EMOTION_CLASS_NAMES):
        return "?", confidence

    return EMOTION_CLASS_NAMES[class_id], confidence

def crop_padded_face(frame, x, y, w, h, padding=40):
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, frame.shape[1])
    y2 = min(y + h + padding, frame.shape[0])
    return frame[y1:y2, x1:x2]

def predict_anti_spoof(face_img, anti_spoof_model):
    if face_img.size == 0:
        return "unknown", 0.0, False

    results = anti_spoof_model.predict(
        source=face_img,
        imgsz=ANTI_SPOOF_IMG_SIZE,
        verbose=False
    )

    probs = results[0].probs
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    class_name = anti_spoof_model.names[class_id].lower()
    is_spoof = class_name == "spoof"

    return class_name, confidence, is_spoof

# load faces from the database and compute their embeddings
def load_faces(db_path, model):
    faces = []
    for name in os.listdir(db_path):
        person_folder = os.path.join(db_path, name)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            print(f"Processing {img_name} for {name}...")
            img_path = os.path.join(person_folder, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            embedding = embed(img_path, model)
            if embedding is not None:
                faces.append({"name": name, "embedding": embedding, "file": img_path})
    return faces


# find the best match for a given embedding among the loaded faces
def find_best_match(embedding, faces, threshold=THRESHOLD):
    best_name = "Unknown"
    best_distance = float('inf')
    for face in faces:
        dist = embedding_distance(embedding, face["embedding"])
        if dist < best_distance:
            best_distance = dist
            best_name = face["name"]

    if best_distance > threshold:
        return "Unknown", best_distance
    return best_name, best_distance

# write recognition events to CSV file
def write_to_csv(row, filepath, headers):
  file_exists = Path(filepath).exists()

  with open(filepath, "a", newline="") as f:
    writer = csv.writer(f)

    # Write header only once
    if not file_exists:
      writer.writerow(headers)

    writer.writerow(row)

def recognition_worker(frame_queue, result_queue, stop_event, model, anti_spoof_model, face_cascade, faces):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue

        results = []

        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE
            )

            for (x, y, w, h) in detected_faces:
                anti_spoof_face = crop_padded_face(frame, x, y, w, h)
                spoof_label, spoof_confidence, is_spoof = predict_anti_spoof(anti_spoof_face, anti_spoof_model)
                face_img = frame[y:y+h, x:x+w]

                if face_img.size == 0:
                    continue

                if is_spoof:
                    results.append({
                        "name": "Spoof",
                        "distance": None,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "face_img": face_img.copy(),
                        "emotion": "?",
                        "spoof_label": spoof_label,
                        "spoof_confidence": spoof_confidence,
                        "is_spoof": True
                    })
                    continue

                face_embedding = embed(face_img, model)

                if face_embedding is None:
                    continue

                name, distance = find_best_match(face_embedding, faces)

                results.append({
                    "name": name,
                    "distance": distance,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "face_img": face_img.copy(),
                    "emotion": "?",
                    "spoof_label": spoof_label,
                    "spoof_confidence": spoof_confidence,
                    "is_spoof": False
                })

        except Exception as e:
            print(f"Error processing frame: {e}")

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except Empty:
                pass

        result_queue.put(results)

def emotion_worker(frame_queue, result_queue, stop_event, emotion_model, face_cascade):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue

        emotion_result = {"emotion": "?", "confidence": 0.0}

        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE
            )

            if len(detected_faces) > 0:
                x, y, w, h = max(detected_faces, key=lambda face: face[2] * face[3])
                face_img = frame[y:y+h, x:x+w]
                emotion, confidence = predict_emotion(face_img, emotion_model)
                emotion_result = {"emotion": emotion, "confidence": confidence}

        except Exception as e:
            print(f"Error detecting emotion: {e}")

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except Empty:
                pass

        result_queue.put(emotion_result)

def main():
    model_path = resolve_model_path()
    print(f"Loading face embedding model from {model_path}...")
    model = load_embedding_model(model_path)

    emotion_model_path = resolve_emotion_model_path()
    print(f"Loading emotion model from {emotion_model_path}...")
    emotion_model = keras.models.load_model(emotion_model_path, compile=False)

    anti_spoof_model_path = resolve_anti_spoof_model_path()
    print(f"Loading anti-spoofing model from {anti_spoof_model_path}...")
    anti_spoof_model = load_anti_spoof_model(anti_spoof_model_path)

    face_cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print(f"Could not load OpenCV face detector from {face_cascade_path}")
        return
    emotion_face_cascade = cv.CascadeClassifier(face_cascade_path)
    if emotion_face_cascade.empty():
        print(f"Could not load OpenCV emotion face detector from {face_cascade_path}")
        return

    print("Loading faces from database...")
    faces = load_faces(DB_PATH, model)
    print(f"Loaded {len(faces)} faces from database.")

    if len(faces) == 0:
        print("No faces found in database. Exiting.")
        return

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)
    emotion_frame_queue = Queue(maxsize=1)
    emotion_result_queue = Queue(maxsize=1)
    stop_event = Event()

    worker = Thread(
        target=recognition_worker,
        args=(frame_queue, result_queue, stop_event, model, anti_spoof_model, face_cascade, faces),
        daemon=True
    )
    worker.start()
    emotion_thread = Thread(
        target=emotion_worker,
        args=(emotion_frame_queue, emotion_result_queue, stop_event, emotion_model, emotion_face_cascade),
        daemon=True
    )
    emotion_thread.start()

    last_results = []
    last_emotion = {"emotion": "?", "confidence": 0.0}
    last_unknown_face = None
    spoof_detected = False
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        display_frame = frame.copy()

        

        # Send a frame to the worker every PROCESS_FRAMES frames
        if frame_count % PROCESS_FRAMES == 0:
            frame_for_workers = frame.copy()
            if frame_queue.empty():
                frame_queue.put(frame_for_workers)
            if emotion_frame_queue.empty():
                emotion_frame_queue.put(frame_for_workers.copy())

        # Collect latest recognition results from worker
        try:
            last_results = result_queue.get_nowait()
        except Empty:
            pass
        try:
            last_emotion = emotion_result_queue.get_nowait()
        except Empty:
            pass

        last_unknown_face = None
        spoof_detected = False

        for res in last_results:
            name = res["name"]
            distance = res["distance"]
            x, y, w, h = res["x"], res["y"], res["w"], res["h"]

            if res.get("is_spoof"):
                color = (0, 0, 255)
                spoof_detected = True
                label = f"SPOOF {res['spoof_confidence']:.2f}"
            elif name == "Unknown":
                color = (0, 0, 255)
                last_unknown_face = res["face_img"]
            else:
                color = (0, 255, 0)

            if not res.get("is_spoof"):
                label = f"{name} ({distance:.2f}) REAL {res['spoof_confidence']:.2f}"

            cv.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

            cv.putText(display_frame, label,
                (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Latest emotion result in top-right corner
        height, width = display_frame.shape[:2]

        box_w = 220
        box_h = 45
        margin = 10

        x1 = width - box_w - margin
        y1 = margin
        x2 = width - margin
        y2 = y1 + box_h

        cv.rectangle(display_frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
        cv.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        emotion_label = f"Emotion: {last_emotion['emotion']}"
        if last_emotion["confidence"] > 0:
            emotion_label += f" {last_emotion['confidence']:.2f}"

        cv.putText(display_frame, emotion_label,
            (x1 + 10, y1 + 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2)
            
        height, width = display_frame.shape[:2]
        cv.rectangle(display_frame, (0, height - 40), (width, height), (30, 30, 30), -1)

        status = "Press Q to quit"
        if spoof_detected:
            status += " | Spoof detected - cannot add"
        if last_unknown_face is not None:
            status += " | Unknown face detected - press A to add"

        cv.putText(display_frame, status, (10, height - 13),
           cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv.imshow("Real-time Face Recognition", display_frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        
        if key == ord('a') and last_unknown_face is not None:
            person_name = input("Enter name for unknown person: ").strip()

            if person_name:
                person_dir = Path(DB_PATH) / person_name
                person_dir.mkdir(parents=True, exist_ok=True)

                filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
                save_path = person_dir / filename

                cv.imwrite(str(save_path), last_unknown_face)

                new_embedding = embed(last_unknown_face, model)

                if new_embedding is not None:
                    faces.append({
                    "name": person_name,
                    "embedding": new_embedding,
                    "file": str(save_path)
                })

                print(f"Added {person_name} to database.")
        frame_count += 1

    stop_event.set()
    worker.join(timeout=1)
    emotion_thread.join(timeout=1)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
