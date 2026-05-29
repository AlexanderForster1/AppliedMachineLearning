from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2 as cv

from gradcam import GradCAM


# How many consecutive frames the new emotion must persist before we log it.
STABILITY_FRAMES = 3

# Hard cap on the number of saved captures. When exceeded, oldest are deleted.
MAX_CAPTURES = 10


class EmotionLogger:
    def __init__(self, emotion_model, log_dir):
        self.gradcam = GradCAM(emotion_model)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # The emotion currently displayed
        self.current_emotion = None

        self.pending_emotion = None
        self.pending_count = 0

        self.lock = Lock()

    def observe(self, frame_bgr, face_box, preprocessed_face,
                emotion, confidence):
        
        # Ignore frames where no usable face was found
        if face_box is None or preprocessed_face is None or emotion in (None, "?"):
            return None

        with self.lock:
            # First-ever stable emotion: just set it without logging
            if self.current_emotion is None:
                self.current_emotion = emotion
                self.pending_emotion = None
                self.pending_count = 0
                return None

            # Same as the displayed emotion: reset any pending change
            if emotion == self.current_emotion:
                self.pending_emotion = None
                self.pending_count = 0
                return None

            # A different emotion this frame: count toward stability
            if emotion == self.pending_emotion:
                self.pending_count += 1
            else:
                self.pending_emotion = emotion
                self.pending_count = 1

            # Not held for long enough yet
            if self.pending_count < STABILITY_FRAMES:
                return None

            # Stable change confirmed: log it.
            saved_path = self._capture(
                frame_bgr, face_box, preprocessed_face, emotion, confidence
            )
            self.current_emotion = emotion
            self.pending_emotion = None
            self.pending_count = 0
            return saved_path

    def _capture(self, frame_bgr, face_box, preprocessed_face,
                 emotion, confidence):
        
        overlay_img = self.gradcam.overlay(
            frame_bgr, face_box, preprocessed_face
        )

        # Annotate the saved image with the predicted emotion + confidence
        label = f"{emotion} ({confidence * 100:.0f}%)"
        cv.putText(overlay_img, label, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"{timestamp}.png"

        # Handle the (rare) case of multiple captures in the same second
        if filename.exists():
            i = 1
            while True:
                candidate = self.log_dir / f"{timestamp}_{i}.png"
                if not candidate.exists():
                    filename = candidate
                    break
                i += 1

        cv.imwrite(str(filename), overlay_img)
        self._enforce_cap()
        return filename

    def _enforce_cap(self):
        captures = sorted(
            self.log_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(captures) - MAX_CAPTURES
        for old in captures[:max(excess, 0)]:
            try:
                old.unlink()
            except OSError:
                pass
