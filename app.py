from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
import time

import cv2 as cv
from flask import Flask, Response, jsonify, render_template, request
from tensorflow import keras

import GUI


app = Flask(__name__)


def json_float(value):
    if value is None:
        return None
    return float(value)


def json_bool(value):
    return bool(value)


def json_text(value):
    if value is None:
        return None
    return str(value)


class CameraProcessor:
    def __init__(self):
        self.lock = Lock()
        self.stop_event = Event()
        self.cap = None
        self.capture_thread = None
        self.inference_thread = None
        self.emotion_thread = None

        self.latest_frame = None
        self.latest_unknown_face = None
        self.latest_detected_face = None
        self.latest_detected_name = None
        self.latest_results = []
        self.latest_emotion = {"emotion": "?", "confidence": 0.0}
        self.latest_status = {
            "running": False,
            "faces": [],
            "emotion": self.latest_emotion,
            "can_add_face": False,
            "can_login": False,
            "detected_person": None,
            "message": "Camera not started",
        }

        self.face_model = None
        self.emotion_model = None
        self.anti_spoof_model = None
        self.face_cascade = None
        self.emotion_face_cascade = None
        self.faces = []

    def load(self):
        if self.face_model is not None:
            return

        model_path = GUI.resolve_model_path()
        print(f"Loading face embedding model from {model_path}...")
        self.face_model = GUI.load_embedding_model(model_path)

        emotion_model_path = GUI.resolve_emotion_model_path()
        print(f"Loading emotion model from {emotion_model_path}...")
        self.emotion_model = keras.models.load_model(emotion_model_path, compile=False)

        anti_spoof_model_path = GUI.resolve_anti_spoof_model_path()
        print(f"Loading anti-spoofing model from {anti_spoof_model_path}...")
        self.anti_spoof_model = GUI.load_anti_spoof_model(anti_spoof_model_path)

        face_cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load OpenCV face detector from {face_cascade_path}")
        self.emotion_face_cascade = cv.CascadeClassifier(face_cascade_path)
        if self.emotion_face_cascade.empty():
            raise RuntimeError(f"Could not load OpenCV emotion face detector from {face_cascade_path}")

        print("Loading faces from database...")
        self.faces = GUI.load_faces(GUI.DB_PATH, self.face_model)
        print(f"Loaded {len(self.faces)} faces from database.")

    def start(self):
        self.load()

        if self.cap is None:
            self.cap = cv.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError("Cannot open camera")

            # Keep the camera buffer tiny so the UI shows recent frames, not stale frames.
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.capture_thread = Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()

        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = Thread(target=self.inference_loop, daemon=True)
            self.inference_thread.start()

        if self.emotion_thread is None or not self.emotion_thread.is_alive():
            self.emotion_thread = Thread(target=self.emotion_loop, daemon=True)
            self.emotion_thread.start()

    def capture_loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_status["running"] = True

    def inference_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.03)
                continue

            results, unknown_face, detected_face, detected_name, message = self.run_inference(frame)

            with self.lock:
                self.latest_results = results
                self.latest_unknown_face = unknown_face
                self.latest_detected_face = detected_face
                self.latest_detected_name = detected_name
                self.latest_status = {
                    "running": True,
                    "faces": self.status_faces(results),
                    "emotion": self.latest_emotion,
                    "can_add_face": unknown_face is not None,
                    "can_login": detected_face is not None and detected_name not in (None, "Unknown", "Spoof"),
                    "detected_person": detected_name,
                    "message": message,
                }

            time.sleep(0.02)

    def emotion_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.03)
                continue

            emotion = self.run_emotion_detection(frame)

            with self.lock:
                self.latest_emotion = emotion
                self.latest_status["emotion"] = emotion

            time.sleep(0.05)

    def run_inference(self, frame):
        results = []
        unknown_face = None
        detected_face = None
        detected_name = None
        message = ""

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=GUI.MIN_FACE_SIZE,
        )

        for (x, y, w, h) in sorted(detected_faces, key=lambda face: face[2] * face[3], reverse=True):
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            anti_spoof_face = GUI.crop_padded_face(frame, x, y, w, h)
            spoof_label, spoof_confidence, is_spoof = GUI.predict_anti_spoof(
                anti_spoof_face,
                self.anti_spoof_model,
            )

            name = "Spoof" if is_spoof else "Unknown"
            distance = None

            if is_spoof:
                message = "Spoof faces cannot be added"
            else:
                face_embedding = GUI.embed(face_img, self.face_model)
                if face_embedding is not None:
                    with self.lock:
                        known_faces = list(self.faces)
                    name, distance = GUI.find_best_match(face_embedding, known_faces)
                    if name == "Unknown":
                        unknown_face = face_img.copy()
                    if detected_face is None:
                        detected_face = face_img.copy()
                        detected_name = name

            results.append({
                "name": json_text(name),
                "distance": json_float(distance),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "anti_spoof": json_text(spoof_label),
                "anti_spoof_confidence": json_float(spoof_confidence),
                "is_spoof": json_bool(is_spoof),
            })

        return results, unknown_face, detected_face, detected_name, message

    def run_emotion_detection(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detected_faces = self.emotion_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=GUI.MIN_FACE_SIZE,
        )

        if len(detected_faces) == 0:
            return {"emotion": "?", "confidence": 0.0}

        x, y, w, h = max(detected_faces, key=lambda face: face[2] * face[3])
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            return {"emotion": "?", "confidence": 0.0}

        emotion, confidence = GUI.predict_emotion(face_img, self.emotion_model)
        return {"emotion": json_text(emotion), "confidence": json_float(confidence)}

    def status_faces(self, results):
        return [{
            "name": json_text(result["name"]),
            "distance": json_float(result["distance"]),
            "anti_spoof": json_text(result["anti_spoof"]),
            "anti_spoof_confidence": json_float(result["anti_spoof_confidence"]),
            "is_spoof": json_bool(result["is_spoof"]),
        } for result in results]

    def draw_overlay(self, frame):
        with self.lock:
            results = list(self.latest_results)

        output = frame.copy()

        for result in results:
            x = result["x"]
            y = result["y"]
            w = result["w"]
            h = result["h"]
            is_spoof = result["is_spoof"]
            name = result["name"]
            distance = result["distance"]
            confidence = result["anti_spoof_confidence"]

            color = (0, 0, 255) if is_spoof or name == "Unknown" else (0, 255, 0)
            label = f"SPOOF {confidence:.2f}" if is_spoof else f"{name} REAL {confidence:.2f}"
            if distance is not None:
                label = f"{name} ({distance:.2f}) REAL {confidence:.2f}"

            cv.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv.putText(output, label, (x, max(y - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return output

    def frames(self):
        self.start()

        while True:
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.03)
                continue

            output = self.draw_overlay(frame)
            ok, buffer = cv.imencode(".jpg", output, [int(cv.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            time.sleep(0.01)

    def add_face(self, person_name):
        person_name = person_name.strip()
        if not person_name:
            return False, "Enter a name."

        with self.lock:
            if self.latest_unknown_face is None:
                return False, "No real unknown face is available to add."
            face_img = self.latest_unknown_face.copy()

        ok, message = self.save_face_for_person(person_name, face_img)
        if not ok:
            return False, message

        with self.lock:
            self.latest_unknown_face = None
            self.latest_status["can_add_face"] = False
            self.latest_status["message"] = f"Added {person_name} to the database."

        return True, f"Added {person_name}."

    def save_face_for_person(self, person_name, face_img):
        person_dir = Path(GUI.DB_PATH) / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        save_path = person_dir / datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        cv.imwrite(str(save_path), face_img)

        new_embedding = GUI.embed(face_img, self.face_model)
        if new_embedding is None:
            return False, "Could not create an embedding for that face."

        with self.lock:
            self.faces.append({
                "name": person_name,
                "embedding": new_embedding,
                "file": str(save_path),
            })

        return True, f"Saved face for {person_name}."

    def login(self):
        with self.lock:
            detected_name = self.latest_detected_name
            can_login = self.latest_status.get("can_login", False)

        if not can_login:
            return False, "No recognized employee is available to log in."

        return True, f"Logged in as {detected_name}."

    def correct_login(self, person_name):
        person_name = person_name.strip()
        if not person_name:
            return False, "Enter the correct name."

        with self.lock:
            if self.latest_detected_face is None:
                return False, "No real detected face is available to correct."
            face_img = self.latest_detected_face.copy()

        ok, message = self.save_face_for_person(person_name, face_img)
        if ok:
            with self.lock:
                self.latest_status["message"] = f"Saved correction for {person_name}."
        return ok, message

    def status(self):
        with self.lock:
            return {
                "running": json_bool(self.latest_status.get("running", False)),
                "faces": self.status_faces(self.latest_status.get("faces", [])),
                "emotion": {
                    "emotion": json_text(self.latest_status.get("emotion", {}).get("emotion", "?")),
                    "confidence": json_float(self.latest_status.get("emotion", {}).get("confidence", 0.0)),
                },
                "can_add_face": json_bool(self.latest_status.get("can_add_face", False)),
                "can_login": json_bool(self.latest_status.get("can_login", False)),
                "detected_person": json_text(self.latest_status.get("detected_person")),
                "message": json_text(self.latest_status.get("message", "")) or "",
                "known_people": sorted({face["name"] for face in self.faces}),
            }


processor = CameraProcessor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        processor.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def status():
    return jsonify(processor.status())


@app.route("/api/add-face", methods=["POST"])
def add_face():
    data = request.get_json(silent=True) or {}
    ok, message = processor.add_face(data.get("name", ""))
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400


@app.route("/api/login", methods=["POST"])
def login():
    ok, message = processor.login()
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400


@app.route("/api/correct-login", methods=["POST"])
def correct_login():
    data = request.get_json(silent=True) or {}
    ok, message = processor.correct_login(data.get("name", ""))
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
