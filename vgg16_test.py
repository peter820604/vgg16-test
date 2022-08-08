import keras
from keras.models import load_model
from keras.models import Sequential

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
# import cv2
import numpy as np
from keras.models import model_from_json
import h5py
import os

# model = Sequential()
modelVGG = VGG16(weights='imagenet', include_top=False)

model = load_model('crossing-model-base-VGG16.h5')

num_test = 1

count = 0
for i in range(num_test):
    # path='pic/'+str(i)+'.png';
    # path='17.png'
    # path='data/validation/veh/'+str(i)+'.png'
    path = r'C:\Users\peter\Desktop\erer.jpg'
    img = load_img(path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = modelVGG.predict(x)
    # print i
    # print i,modelMY.predict_proba(features)
    if (model.predict_proba(features) > 0.7):
        print(i)

print(count * 1.0 / num_test)
# print '\n'
