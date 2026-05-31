import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config error: {e}")

from cnnbackbone import model as base_cnn


def load_triplets_from_csv(filename):
    anchors, positives, negatives = [], [], []
    if not os.path.exists(filename):
        print(f"Err")
        return [], [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 3:
                anchors.append(row[0])
                positives.append(row[1])
                negatives.append(row[2])
    return anchors, positives, negatives


def load_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    return tf.cast(img, tf.float32) / 255.0


def load_triplet(anchor_path, positive_path, negative_path):
    return (
        load_image(anchor_path),
        load_image(positive_path),
        load_image(negative_path)
    )


def create_dataset(anchors, positives, negatives, batch_size):
    anchor_ds = tf.data.Dataset.from_tensor_slices(anchors)
    positive_ds = tf.data.Dataset.from_tensor_slices(positives)
    negative_ds = tf.data.Dataset.from_tensor_slices(negatives)

    dataset = tf.data.Dataset.zip((anchor_ds, positive_ds, negative_ds))
    dataset = dataset.shuffle(buffer_size=min(len(anchors), 10000))
    dataset = dataset.map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def build_siamese_model(base_cnn):
    anchor_input = layers.Input(name="anchor", shape=(64, 64, 3))
    positive_input = layers.Input(name="positive", shape=(64, 64, 3))
    negative_input = layers.Input(name="negative", shape=(64, 64, 3))

    distances = DistanceLayer()(
        base_cnn(anchor_input),
        base_cnn(positive_input),
        base_cnn(negative_input),
    )
    return models.Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )


class SiameseModel(models.Model):
    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]


if __name__ == "__main__":
    train_anchors, train_positives, train_negatives = load_triplets_from_csv('train_triplets.csv')
    if not train_anchors:
        import sys
        sys.exit(1)

    val_anchors, val_positives, val_negatives = load_triplets_from_csv('val_triplets.csv')

    train_dataset = create_dataset(train_anchors, train_positives, train_negatives, batch_size=300)
    val_dataset = create_dataset(val_anchors, val_positives, val_negatives, batch_size=300)

    siamese_network = build_siamese_model(base_cnn)
    siamese_model = SiameseModel(siamese_network, margin=0.8)

    siamese_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,       
        patience=1,       
        min_lr=1e-6,      
        verbose=1         
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    epochs = 8
    history = siamese_model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[lr_schedule, early_stopping]
    )

    base_cnn.save("face_embedding_modeleuclidean.keras")

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Siamese Network Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_losseuclidean.png')
