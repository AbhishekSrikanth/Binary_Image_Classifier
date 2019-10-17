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

#Add layers from VGG16 except last layer
for layer in vgg16_model.layers[:-1]:
    
    model.add(layer)
    
for layer in model.layers:
    
    #Freeze added layers which are already trained
    layer.trainable = False

#Add new dense layer for prediction and fine tuning
model.add(Dense(1,activation='sigmoid'))

#Load model weights from diretory 
model.load_weights('')

#Compile the model with loaded weights
model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

#Set the confidence threshold with which label is decided
confidence_Threshold = 0.5

cam = cv2.VideoCapture(0)

#Create new window 
cv2.namedWindow("CameraFeed")

#Loop until key press
while True:

    #Get frame from the camera feed
    ret, frame = cam.read()

    #Resize frame size to fit model input size
    im = cv2.resize(frame, (224, 224))
    im = np.expand_dims(im, axis=0)

    #Make prediction using the model
    prediction = model.predict(im)

    #If model is confident predict smoke
    #Frames with Non Smoke label would be predominant  
    if prediction[0][0] <= confidence_Threshold:
        label = 'No Smoke'
    else:
        label = 'Smoke'

    #Text Label attributes
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    TopLeftCorner          = (10,25)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2

    #Add label to the frame
    cv2.putText(frame,label, 
    TopLeftCorner, 
    font, 
    fontScale,
    fontColor,
    lineType)

    #Display the frame in the created window
    cv2.imshow("CameraFeed", frame)

    if not ret:
        break

    #Wait for keypress
    k = cv2.waitKey(1)

    #Detect ESC key 
    if k%256 == 27:

        # ESC pressed
        print("Closing Window")
        break

#Release created window
cam.release()

cv2.destroyAllWindows()
