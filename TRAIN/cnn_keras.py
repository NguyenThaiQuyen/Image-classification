import numpy as np
import tensorflow
from keras.models import model_from_json
from keras import layers
from tensorflow import keras
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.pyplot import imshow

def AlexNet(input_shape, number_classes):
    
    X_input = Input(input_shape)
    
    X = Conv2D(96,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)
    
    X = Conv2D(256,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)
    
    X = Conv2D(384, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(384, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)
    
    X = Flatten()(X)
    
    X = Dense(4096, activation = 'relu', name = "fc0")(X)
    
    X = Dense(4096, activation = 'relu', name = 'fc1')(X) 
    
    X = Dense(number_classes,activation='softmax',name = 'fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='AlexNet')
    return model

number_classes = 3
alex = AlexNet((227,227,3), number_classes)
alex.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics=['accuracy'])


#  get data from folder and data augmented
path_dir_data = "./DATA/train"
train_datagen = ImageDataGenerator(rescale=1. /255)
train = train_datagen.flow_from_directory(path_dir_data, target_size=(227,227), class_mode='categorical')

# train model
alex.fit(train,epochs=50)
alex.save("./MODEL")

# get evaluate model
path_validation = '/content/drive/My Drive/DATA/validation'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test = test_datagen.flow_from_directory(path_validation, target_size=(227,227), class_mode='categorical')
scores = alex.evaluate(test)
print(scores)

# load model saved
class_names = ['elephant', 'rabbit', 'tiger']

model = keras.models.load_model("./MODEL")

# get an image
from google.colab import files
uploaded = files.upload()
new_image = plt.imread(next(iter(uploaded)))

resized_image = resize(new_image, (227,227,3))

input_arr = np.array([resized_image])  # Convert single image to a batch

# predict
predictions = model.predict(input_arr)

for i in range(len(predictions[0])):
  predictions[0][i] = round(predictions[0][i],7)
result = predictions[0]
index_class_result = np.where(result == np.amax(result))[0][0]
print(class_names[index_class_result])