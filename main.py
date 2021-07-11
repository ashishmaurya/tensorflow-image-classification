import tensorflow as tf
from tensorflow import keras

# Load images in dataset (so that we don't have to load all imagees in memory at once)
dataset = keras.preprocessing.image_dataset_from_directory(
    './data/train_py', image_size=(1920, 1080),batch_size=1)

# create model
model = keras.Sequential()
# add preprocessing layer to reduce the image size(to reduce computation complexity)
model.add(tf.keras.layers.experimental.preprocessing.Resizing(640,360))
# add Convulation Layer(good for image classification)
model.add(tf.keras.layers.Conv2D(
    128, 3, activation='sigmoid', input_shape=(640, 360, 3),batch_size=1))
# Flatten the output from above layer as softmax accepts vectors(1 dimension)
model.add(tf.keras.layers.Flatten())
# Add softmax layer to convert it into probability
model.add(tf.keras.layers.Dense(2,activation='softmax'))
# Compile  the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train your model
model.fit(dataset,epochs=5)

# Save model onto disk
model.save('./ts-models/py-car-cmd')
