import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from pathlib import Path
train_dir = "data/data/classification_data/train_data"
val_dir = "data/data/classification_data/val_data"
test_dir = "data/data/classification_data/test_data"
pairs_file = "data/data/verification_pairs_val.txt"
dataset_root = "data/data"
print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices("GPU"))

train_ds = image_dataset_from_directory(train_dir, image_size=(150, 150), batch_size=32, label_mode='int')
val_ds = image_dataset_from_directory(val_dir, image_size=(150, 150), batch_size=32, label_mode='int')
test_ds = image_dataset_from_directory(test_dir, image_size=(150, 150), batch_size=32, label_mode='int')
num_of_classes = len(train_ds.class_names)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.04),
    tf.keras.layers.RandomZoom(0.08),
])
model = models.Sequential()
model.add(tf.keras.Input(shape=(150, 150, 3)))
model.add(data_augmentation)
model.add(layers.Rescaling(1./255,))
model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, name="embedding"))
model.add(layers.Dense(num_of_classes, activation='softmax', name="classifier"))
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=75, callbacks=[early_stopping, reduce_lr])
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")


'''
----------------------------------------------------
ROC curve and AUC for face verification models
----------------------------------------------------

'''
embedding_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("embedding").output)
print("saving models")
embedding_model.save("face_embedding_model.keras")
model.save("Supervised_face_model.keras")
import pandas as pd

def load_image(image_path, image_size=(150, 150)):
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    img = tf.keras.utils.img_to_array(img)
    return img

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
same_distances = distances[labels == 1]
different_distances = distances[labels == 0]

plt.figure(figsize=(8, 5))
plt.hist(same_distances, bins=40, alpha=0.6, label="Same person")
plt.hist(different_distances, bins=40, alpha=0.6, label="Different person")
plt.xlabel("Euclidean Distance")
plt.ylabel("Frequency")
plt.title("Embedding Distance Distribution")
plt.legend()
plt.grid(True)
plt.show()
# Euclidean AUC
euclidean_scores = -distances
fpr_euc, tpr_euc, _ = roc_curve(labels, euclidean_scores)
auc_euc = auc(fpr_euc, tpr_euc)

# Cosine AUC
cosine_scores = np.sum(emb_a * emb_b, axis=1)
fpr_cos, tpr_cos, _ = roc_curve(labels, cosine_scores)
auc_cos = auc(fpr_cos, tpr_cos)

print("Euclidean AUC:", auc_euc)
print("Cosine AUC:", auc_cos)

plt.figure(figsize=(7, 6))
plt.plot(fpr_euc, tpr_euc, label=f"ROC curve, AUC = {roc_euc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face Verification ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

j_scores = tpr_euc - fpr_euc

best_index = np.argmax(tpr_euc - fpr_euc)
best_similarity_threshold = thresholds[best_index]
predictions = (similarity_scores >= best_similarity_threshold).astype(int)


print("Best distance threshold:", best_similarity_threshold)

acc = accuracy_score(labels, predictions)
cm = confusion_matrix(labels, predictions)

print("Verification accuracy:", acc)
print("Confusion matrix:")
print(cm)

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")

if "val_accuracy" in history.history:
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")

if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve, AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Face Verification ROC Curve")
plt.legend()
plt.grid(True)
plt.show()


