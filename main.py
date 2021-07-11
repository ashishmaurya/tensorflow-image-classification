import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras.backend import conv2d

dataset = keras.preprocessing.image_dataset_from_directory(
    './data/train_py', image_size=(1920, 1080),batch_size=1)
# size = (200, 200)
# dataset = dataset.map(lambda img: tf.image.resize(img, size))
model = keras.Sequential()
model.add(tf.keras.layers.experimental.preprocessing.Resizing(640,360))
model.add(tf.keras.layers.Conv2D(
    128, 3, activation='sigmoid', input_shape=(640, 360, 3),batch_size=1))
# model.add(tf.keras.layers.MaxPool2D())
# model.add(tf.keras.layers.Dropout(0.4))
# model.add(tf.keras.layers.Conv2D(
#     64, 3, activation='sigmoid', input_shape=(640, 360, 3),batch_size=1))
# model.add(tf.keras.layers.MaxPool2D())
# model.add(tf.keras.layers.Dropout(0.4))
# model.add(tf.keras.layers.Conv2D(
#     32, 3, activation='sigmoid', input_shape=(640, 360, 3),batch_size=1))
# model.add(tf.keras.layers.MaxPool2D())
# model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2,activation='softmax'))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# model.summary()
model.fit(dataset,epochs=5)

model.save('./ts-models/py-car-cmd')

# model.evaluate(dataset)