from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
import os
import time

import cv2 as cv
from flask import Flask, Response, jsonify, render_template, request
from tensorflow import keras

import GUI


app = Flask(__name__)

# Helper functions to convert values to JSON-serializable formats for the API responses.
def json_float(value):
    if value is None:
        return None
    return float(value)

# Convert a value to a boolean for JSON serialization.
def json_bool(value):
    return bool(value)

# Convert a value to a string for JSON serialization, handling None as null.
def json_text(value):
    if value is None:
        return None
    return str(value)


def video_capture(source):
    if os.name == "nt" and isinstance(source, int):
        return cv.VideoCapture(source, cv.CAP_DSHOW)
    return cv.VideoCapture(source)

# The main class that handles camera capture, face recognition, emotion detection, and API interactions.
class CameraProcessor:
    def __init__(self):
        self.lock = Lock()
        self.capture_lock = Lock()
        self.stop_event = Event()
        self.cap = None
        self.camera_source = 0
        self.max_camera_index = 5
        self.camera_sources_cache = None
        self.camera_sources_checked_at = 0
        self.camera_sources_ttl = 30
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
            "camera_source": self.camera_source,
            "camera_sources": [],
            "message": "Camera not started",
        }

        self.face_model = None
        self.emotion_model = None
        self.anti_spoof_model = None
        self.face_cascade = None
        self.emotion_face_cascade = None
        self.faces = []

    # Load the face embedding model, emotion detection model, anti-spoofing model, and Haar cascades for face detection. 
    # Also load known faces from the database.
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

    def parse_camera_source(self, source):
        if isinstance(source, int):
            return source
        source = str(source).strip()
        if source.isdigit():
            return int(source)
        return 0

    def discover_camera_sources(self, force=False):
        now = time.time()
        if (
            not force
            and self.camera_sources_cache is not None
            and now - self.camera_sources_checked_at < self.camera_sources_ttl
        ):
            return list(self.camera_sources_cache)

        sources = []

        with self.capture_lock:
            current_source = self.camera_source
            current_cap = self.cap

        for index in range(self.max_camera_index + 1):
            if index == current_source and current_cap is not None and current_cap.isOpened():
                sources.append({"id": index, "label": f"Camera {index}"})
                continue

            cap = video_capture(index)
            if cap.isOpened():
                sources.append({"id": index, "label": f"Camera {index}"})
            cap.release()

        if not sources:
            sources.append({"id": current_source, "label": f"Camera {current_source}"})

        self.camera_sources_cache = list(sources)
        self.camera_sources_checked_at = now
        return sources

    def open_camera(self, source):
        parsed_source = self.parse_camera_source(source)
        cap = video_capture(parsed_source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera source: {parsed_source}")

        # Keep the camera buffer tiny so the UI shows recent frames, not stale frames.
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        with self.capture_lock:
            old_cap = self.cap
            self.cap = cap
            self.camera_source = parsed_source

        if old_cap is not None:
            old_cap.release()

        with self.lock:
            self.latest_frame = None
            self.latest_unknown_face = None
            self.latest_detected_face = None
            self.latest_detected_name = None
            self.latest_results = []
            self.latest_emotion = {"emotion": "?", "confidence": 0.0}
            self.latest_status = {
                "running": True,
                "faces": [],
                "emotion": self.latest_emotion,
                "can_add_face": False,
                "can_login": False,
                "detected_person": None,
                "camera_source": self.camera_source,
                "message": f"Camera source changed to {self.camera_source}",
            }

    # Start the camera capture and processing threads if they are not already running.
    def start(self):
        self.load()

        with self.capture_lock:
            needs_camera = self.cap is None

        if needs_camera:
            self.open_camera(self.camera_source)

        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.capture_thread = Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()

        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = Thread(target=self.inference_loop, daemon=True)
            self.inference_thread.start()

        if self.emotion_thread is None or not self.emotion_thread.is_alive():
            self.emotion_thread = Thread(target=self.emotion_loop, daemon=True)
            self.emotion_thread.start()

    # The main loop for capturing frames from the camera. 
    # It continuously reads frames and updates the latest frame and status.
    def capture_loop(self):
        while not self.stop_event.is_set():
            with self.capture_lock:
                cap = self.cap
                ok, frame = (False, None) if cap is None else cap.read()

            if not ok:
                time.sleep(0.03)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_status["running"] = True
                self.latest_status["camera_source"] = self.camera_source

    # The main loop for running face recognition and anti-spoofing inference on the latest captured frame. 
    # It updates the latest results and status based on the inference.
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
                    "camera_source": self.camera_source,
                    "message": message,
                }

            time.sleep(0.02)

    # The main loop for running emotion detection on the latest captured frame. 
    # It updates the latest emotion and status based on the inference.
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

    # Run face detection, recognition, and anti-spoofing inference on the given frame. 
    # It returns the results and any detected unknown or recognized faces.
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

    # Run emotion detection on the given frame. 
    # It detects faces in the frame and predicts the emotion for the largest detected face, returning the emotion and confidence.
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

    # Convert the raw inference results into a format suitable for the API response, 
    # extracting relevant fields and converting them to JSON-serializable types.
    def status_faces(self, results):
        return [{
            "name": json_text(result["name"]),
            "distance": json_float(result["distance"]),
            "anti_spoof": json_text(result["anti_spoof"]),
            "anti_spoof_confidence": json_float(result["anti_spoof_confidence"]),
            "is_spoof": json_bool(result["is_spoof"]),
        } for result in results]

    # Draw rectangles and labels on the given frame based on the latest inference results, 
    # indicating detected faces, their recognized names, distances, and spoofing status.
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

    # A generator function that yields JPEG-encoded frames with overlay for streaming to the web interface. 
    # It continuously captures frames, runs inference, and encodes the output for display.
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

    # Add a new face to the database with the given name, using the latest detected unknown face. 
    # It saves the face image, creates an embedding, and updates the known faces list and status.
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

    # Save the given face image for the specified person name in the database directory. 
    # It creates an embedding for the face and updates the known faces list.
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

    # Attempt to log in using the latest detected face. 
    # It checks if a recognized employee is available and returns the appropriate message.
    def login(self):
        with self.lock:
            detected_name = self.latest_detected_name
            can_login = self.latest_status.get("can_login", False)

        if not can_login:
            return False, "No recognized employee is available to log in."

        return True, f"Logged in as {detected_name}."

    # Correct the login by saving the latest detected face under the provided person name. 
    # This allows users to correct misrecognized faces by adding them to the database with the correct name.
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

    def set_camera_source(self, source):
        parsed_source = self.parse_camera_source(source)
        available_ids = {item["id"] for item in self.discover_camera_sources()}
        if parsed_source not in available_ids:
            message = f"Camera {parsed_source} is not available."
            with self.lock:
                self.latest_status["message"] = message
            return False, message

        try:
            self.open_camera(parsed_source)
        except RuntimeError as e:
            with self.lock:
                self.latest_status["message"] = str(e)
            return False, str(e)

        return True, f"Camera source changed to {self.camera_source}."

    # Get the current status of the system, 
    # including whether it's running, detected faces, emotion, and other relevant information for the API response.
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
                "camera_source": json_text(self.latest_status.get("camera_source", self.camera_source)),
                "camera_sources": list(self.camera_sources_cache or []),
                "message": json_text(self.latest_status.get("message", "")) or "",
                "known_people": sorted({face["name"] for face in self.faces}),
            }


processor = CameraProcessor()

# Flask route for the main page, rendering the index.html template.
@app.route("/")
def index():
    return render_template("index.html")

# Flask route for the video feed, streaming the processed frames with overlay to the web interface.
@app.route("/video_feed")
def video_feed():
    return Response(
        processor.frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# Flask route for the API endpoint to get the current status of the system, 
# returning JSON data with detected faces, emotion, and other relevant information.
@app.route("/api/status")
def status():
    return jsonify(processor.status())

# Flask route for the API endpoint to add a new face to the database. 
# It accepts a POST request with the person's name and uses the latest detected unknown face to create a new entry in the database.
@app.route("/api/add-face", methods=["POST"])
def add_face():
    data = request.get_json(silent=True) or {}
    ok, message = processor.add_face(data.get("name", ""))
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400

# Flask route for the API endpoint to log in using the latest detected face. 
# It checks if a recognized employee is available and returns the appropriate message.
@app.route("/api/login", methods=["POST"])
def login():
    ok, message = processor.login()
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400

# Flask route for the API endpoint to correct a login by saving the latest detected face under the provided person name. 
# This allows users to correct misrecognized faces by adding them to the database with the correct name.
@app.route("/api/correct-login", methods=["POST"])
def correct_login():
    data = request.get_json(silent=True) or {}
    ok, message = processor.correct_login(data.get("name", ""))
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400


@app.route("/api/camera-source", methods=["POST"])
def camera_source():
    data = request.get_json(silent=True) or {}
    ok, message = processor.set_camera_source(data.get("source", 0))
    return jsonify({"ok": ok, "message": message}), 200 if ok else 400


@app.route("/api/camera-sources")
def camera_sources():
    force = request.args.get("refresh") == "1"
    return jsonify({"sources": processor.discover_camera_sources(force=force)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
