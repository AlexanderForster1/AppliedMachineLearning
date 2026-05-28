import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

embedding_model = tf.keras.models.load_model("face_embedding_model.keras", safe_mode=False)

embedding_model.summary()

def load_image(image_path, image_size=(150, 150)):
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    img = tf.keras.utils.img_to_array(img)
    return img

img_a_path = Path("data/data/verification_data/00041961.jpg")
img_b_path = Path("data/data/verification_data/00041962.jpg")
img_c_path = Path("data/data/verification_data/00041963.jpg")

images = np.array([
    load_image(img_a_path),
    load_image(img_b_path),
    load_image(img_c_path)
])

embeddings = embedding_model.predict(images, batch_size=32)

print("Embedding shape:", embeddings.shape)
print("Embedding min:", embeddings.min())
print("Embedding max:", embeddings.max())
print("Embedding mean:", embeddings.mean())
print("Embedding std:", embeddings.std())

print("First embedding first 10 values:")
print(embeddings[0][:10])

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

dist_ab = euclidean_distance(embeddings[0], embeddings[1])
dist_ac = euclidean_distance(embeddings[0], embeddings[2])
dist_bc = euclidean_distance(embeddings[1], embeddings[2])

print("Distance A-B:", dist_ab)
print("Distance A-C:", dist_ac)
print("Distance B-C:", dist_bc)

import pandas as pd

pairs_file = Path("data/data/verification_pairs_val.txt")
base_dir = Path("data/data")

pairs_df = pd.read_csv(
    pairs_file,
    sep=r"\s+",
    header=None,
    names=["image1", "image2", "label"]
)

print(pairs_df.head())
print(pairs_df.columns)

img1_col = "image1"
img2_col = "image2"
label_col = "label"

def load_pair_dataset(pairs_df, base_dir, img1_col, img2_col, label_col, image_size=(150, 150), max_pairs=None):
    if max_pairs is not None:
        pairs_df = pairs_df.sample(n=min(max_pairs, len(pairs_df)), random_state=42)

    images_a = []
    images_b = []
    labels = []

    for _, row in pairs_df.iterrows():
        path_a = base_dir / row[img1_col]
        path_b = base_dir / row[img2_col]

        if not path_a.exists():
            raise FileNotFoundError(f"Missing file A: {path_a}")

        if not path_b.exists():
            raise FileNotFoundError(f"Missing file B: {path_b}")

        images_a.append(load_image(path_a, image_size))
        images_b.append(load_image(path_b, image_size))
        labels.append(int(row[label_col]))

    return np.array(images_a), np.array(images_b), np.array(labels)

pairs_a, pairs_b, labels = load_pair_dataset(
    pairs_df=pairs_df,
    base_dir=base_dir,
    img1_col=img1_col,
    img2_col=img2_col,
    label_col=label_col,
    image_size=(150, 150),
    max_pairs=None
)

print("pairs_a:", pairs_a.shape)
print("pairs_b:", pairs_b.shape)
print("labels:", labels.shape)
print("Label counts:", np.unique(labels, return_counts=True))

def l2_normalize(x, eps=1e-10):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

emb_a = embedding_model.predict(pairs_a, batch_size=32)
emb_b = embedding_model.predict(pairs_b, batch_size=32)

emb_a = l2_normalize(emb_a)
emb_b = l2_normalize(emb_b)

distances = np.linalg.norm(emb_a - emb_b, axis=1)

print("Distance min:", distances.min())
print("Distance max:", distances.max())
print("Distance mean:", distances.mean())
print("Distance std:", distances.std())

print("First 20 distances:", distances[:20])
print("First 20 labels:", labels[:20])

same_distances = distances[labels == 1]
different_distances = distances[labels == 0]

print("Same-person mean distance:", same_distances.mean())
print("Different-person mean distance:", different_distances.mean())

print("Same-person median distance:", np.median(same_distances))
print("Different-person median distance:", np.median(different_distances))

plt.figure(figsize=(8, 5))
plt.hist(same_distances, bins=40, alpha=0.6, label="Same person")
plt.hist(different_distances, bins=40, alpha=0.6, label="Different person")
plt.xlabel("Euclidean Distance")
plt.ylabel("Frequency")
plt.title("Embedding Distance Distribution")
plt.legend()
plt.grid(True)
plt.show()

similarity_scores = -distances
fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC curve, AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face Verification ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

j_scores = tpr - fpr
best_index = np.argmax(j_scores)

best_similarity_threshold = thresholds[best_index]
best_distance_threshold = -best_similarity_threshold

print("Best distance threshold:", best_distance_threshold)

predictions = (distances <= best_distance_threshold).astype(int)

acc = accuracy_score(labels, predictions)
cm = confusion_matrix(labels, predictions)

print("Verification accuracy:", acc)
print("Confusion matrix:")
print(cm)

