import h5py
import os
import cv2
import numpy as np
from predictwithmodel import PredictWithModel
import configs as configs

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
    prediction = PredictWithModel.getPredictions(im)

    #If model is confident predict smoke
    #Frames with Non Smoke label would be predominant  
    if prediction[0][0] <= 0.5:
        label = configs.Label1
    else:
        label = configs.Label2

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