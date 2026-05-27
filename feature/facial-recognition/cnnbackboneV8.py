import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models



model = models.Sequential()

#data augmentation layers
#model.add(layers.RandomFlip("horizontal", input_shape=(64, 64, 3)))
#model.add(layers.RandomRotation(0.05))
model.add(layers.RandomZoom(0.2, input_shape=(64, 64, 3)))
#model.add(layers.RandomBrightness(0.2))

model.add(layers.Conv2D(32, (5, 5), padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256))
model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))


if __name__ == "__main__":
    model.summary()
