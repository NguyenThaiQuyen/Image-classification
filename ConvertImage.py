import json   
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

def converImage(data):
    image = cv2.imdecode(np.frombuffer(bytearray(data), np.uint8), -1)
    resized_image = resize(image, (227,227,3))
    input_arr = np.array([resized_image]) 
    return input_arr