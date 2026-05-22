import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_and_preprocess_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

def verify_faces(img1_path, img2_path, model_path="face_embedding_model.keras", threshold=1.3):

    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"{e}")
        return

    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)

    embedding1 = model.predict(img1, verbose=0)
    embedding2 = model.predict(img2, verbose=0)

    distance = np.sum(np.square(embedding1 - embedding2))

    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    
    if distance < threshold:
        print(f"Prediction: SAME PERSON")
    else:
        print(f"Prediction: DIFFERENT PERSON)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <path_to_image_1> <path_to_image_2>")
    else:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
        threshold = 1.4
        verify_faces(img1, img2, threshold=threshold)
