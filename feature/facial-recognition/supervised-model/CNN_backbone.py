import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
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
model.add(layers.Dense(128))
model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
model.summary()