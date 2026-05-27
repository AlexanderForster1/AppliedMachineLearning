"""
COS30082 ASSIGN1 - EMOTION DETECTION
TIM JOHNSON
STUDENT ID 106013868
"""

# import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

# for use with google colab
# !unzip -q archive.zip

"""# Step 2: Load and prepare dataset"""

# prepare images for training
img_height = 48
img_width = 48
batch_size = 64

# filter out disgusted, fearful, surprised
keep = ['angry', 'happy', 'neutral', 'sad']

def make_filtered_dir(source_dir, filtered_dir, keep):
    os.makedirs(filtered_dir, exist_ok=True)
    for cls in keep:
        src = os.path.abspath(os.path.join(source_dir, cls))
        dst = os.path.join(filtered_dir, cls)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    return filtered_dir

# data paths
train_dir = make_filtered_dir("train", "train_filtered", keep)
test_dir = make_filtered_dir("test", "test_filtered", keep)

# 90/10 split of train into train/val, keep test separate
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)
print("Number of classes:", num_classes)

train_count = tf.data.experimental.cardinality(train_ds).numpy() * batch_size
val_count = tf.data.experimental.cardinality(val_ds).numpy() * batch_size
test_count = tf.data.experimental.cardinality(test_ds).numpy() * batch_size

print("Approx. training samples:", train_count)
print("Approx. validation samples:", val_count)
print("Approx. test samples:", test_count)

# performance optimisation: cache, shuffle, prefetch
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""# Step 4: Build the CNN model"""

# augmentation pipeline - applied only at training time
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(48, 48, 1)),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.35),

    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.6),
    layers.Dense(num_classes, activation='softmax')
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

"""# Step 5: Train the model"""

# compute class weights
# (disgust has ~10x fewer samples than happy, so weight it more)
counts = np.array([len(os.listdir(os.path.join(train_dir, c))) for c in class_names])
total = counts.sum()
class_weights = {i: float(total / (len(counts) * c)) for i, c in enumerate(counts)}
print("Per-class sample counts:")
for name, c in zip(class_names, counts):
    print(f"  {name:12s} {c}")
print("\nClass weights:")
for i, w in class_weights.items():
    print(f"  {class_names[i]:12s} {w:.3f}")

# callbacks: save best, drop LR on plateau, stop early if no improvement
callbacks = [
    ModelCheckpoint(
        "emotion_model.keras",
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=4,
        factor=0.5,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
]

# train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights
)

# evaluate
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)

# accuracy curves
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# loss curves
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# save the model
model.save("emotion_model.keras")
print("Model saved to emotion_model.keras")

# evaluation imports
from sklearn.metrics import confusion_matrix, classification_report

# collect true labels and model predictions over the test set
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds) # softmax probabilities
y_pred = np.argmax(y_prob, axis=1) # predicted class index

print("Test samples:", len(y_true))
print("Probability matrix shape:", y_prob.shape)

# normalised confusion matrix (each row sums to 1 -> recall per class)
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted emotion')
ax.set_ylabel('True emotion')
ax.set_title('Confusion Matrix')

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                color='white' if cm_norm[i, j] > 0.5 else 'black')

plt.tight_layout()
plt.show()

# per-class precision / recall / f1
report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print(report)