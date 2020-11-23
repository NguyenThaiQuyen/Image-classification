import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("E:/db_demo/MODEL", custom_objects=None, compile=True)
# class_names = ['elephant', 'rabbit', 'tiger']
def predict(image_array):
    predictions = model.predict(image_array)
    for i in range(len(predictions[0])):
       predictions[0][i] = round(predictions[0][i],7)
    result = predictions[0]
    index_class_result = np.where(result == np.amax(result))[0][0]

    return index_class_result
