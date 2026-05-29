import cv2 as cv
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        if hasattr(layer, "layers"):
            for inner in reversed(layer.layers):
                if isinstance(inner, tf.keras.layers.Conv2D):
                    return inner
    raise ValueError("No Conv2D layer found in model")


class GradCAM:

    def __init__(self, model, target_layer_name=None):
        self.model = model
        if target_layer_name is None:
            self.target_layer = _find_last_conv_layer(model)
        else:
            self.target_layer = model.get_layer(target_layer_name)
        self.target_layer_name = self.target_layer.name

    def heatmap(self, preprocessed_input, class_index=None):
        captured = {}
        original_call = self.target_layer.call

        def capture_call(*args, **kwargs):
            out = original_call(*args, **kwargs)
            captured["activations"] = out
            return out

        self.target_layer.call = capture_call
        try:
            input_tensor = tf.convert_to_tensor(preprocessed_input)
            with tf.GradientTape() as tape:
                predictions = self.model(input_tensor, training=False)
                conv_outputs = captured["activations"]
                tape.watch(conv_outputs)
                if class_index is None:
                    class_index = int(tf.argmax(predictions[0]))
                class_score = predictions[:, class_index]
            grads = tape.gradient(class_score, conv_outputs)
        finally:
            self.target_layer.call = original_call

        # Global-average-pool the gradients to get per-channel importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        # Weighted combination of feature maps
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalise to [0, 1]
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()

    def overlay(self, frame_bgr, face_box, preprocessed_face,
                class_index=None, alpha=0.45):
        hmap = self.heatmap(preprocessed_face, class_index=class_index)

        x, y, w, h = face_box
        # Resize the small conv-feature heatmap up to the face box size
        hmap_resized = cv.resize(hmap, (w, h))
        # Convert to 8-bit and apply a colour map
        hmap_uint8 = np.uint8(255 * hmap_resized)
        coloured = cv.applyColorMap(hmap_uint8, cv.COLORMAP_JET)

        output = frame_bgr.copy()
        # Blend the coloured heatmap onto the face region only
        face_region = output[y:y + h, x:x + w]
        if face_region.shape[:2] != coloured.shape[:2]:
            # Defensive: skip overlay if shapes mismatch (e.g. box clipped at edge)
            return output
        blended = cv.addWeighted(coloured, alpha, face_region, 1 - alpha, 0)
        output[y:y + h, x:x + w] = blended

        # Draw the face box
        cv.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2)
        return output
