# tensorflow-image-classification
Python Keras Model to build and train image classification for bike and car

`main.py` - used to build the model and save it to disk(so that it can be used latter)

`predict.py` - used to loaded the trained model and try to pridict the sample image

`data` folder is arranged in such was so that `keras.preprocessing.image_dataset_from_directory` function can be used  for loading the data in dataset

### Note : I had installed the `tensorflow-cpu` instead of `tensflow`