import tensorflow as tf
from tensorflow import keras
import numpy as np

# load saved model
model = keras.models.load_model('./ts-models/py-car-cmd')

#Load your own  image to pridict
image = tf.keras.preprocessing.image.load_img('data/test/car_1.jpg', target_size=(1920,1080))

input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
print(predictions)