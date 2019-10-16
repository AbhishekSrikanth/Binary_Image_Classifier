import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Dense, Flatten, ZeroPadding2D, Activation
import h5py
import os
import cv2
import tensorflow as tf
import numpy as np

#Initialize with ImageNet weights
vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet')

#Create new sequential model
model = Sequential()

for layer in vgg16_model.layers[:-1]:
    
    #Add layers from VGG16
    model.add(layer)
    
for layer in model.layers:
    
    #Freeze added layers which are already trained
    layer.trainable = False

#Add new dense layer for prediction
model.add(Dense(1,activation='sigmoid'))

#Load model weights from diretory 
model.load_weights('')

#Compile the model with loaded weights
model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:

    ret, frame = cam.read()

    im = cv2.resize(frame, (224, 224))
    im = np.expand_dims(im, axis=0)
    prediction = model.predict(im)

    #If model is confident predict smoke
    if prediction[0][0] <= 0.5:
        label = 'NoSmoke'
    else:
        label = 'Smoke'

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    TopLeftCorner          = (10,25)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame,label, 
    TopLeftCorner, 
    font, 
    fontScale,
    fontColor,
    lineType)

    cv2.imshow("test", frame)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()
