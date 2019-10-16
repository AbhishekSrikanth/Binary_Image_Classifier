
# coding: utf-8

# In[2]:


import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, ZeroPadding2D, Activation
from keras.preprocessing.image import ImageDataGenerator
import h5py
import os
import cv2
import tensorflow as tf
import numpy as np


# In[3]:


train_data_dir = 'C:/Users/abhis/Pictures/Smoke_Dataset/Train'
valid_data_dir = 'C:/Users/abhis/Pictures/Smoke_Dataset/Valid'

img_width, img_height = 224,224


# In[4]:


vgg16_model = keras.applications.vgg16.VGG16(weights='Z:/vgg16_weights_tf_dim_ordering_tf_kernels.h5')


# In[5]:


vgg16_model.summary()


# In[6]:


model = Sequential()

for layer in vgg16_model.layers[:-1]:
    
    model.add(layer)
    
for layer in model.layers:
    
    layer.trainable = False
    
model.add(Dense(1,activation='sigmoid'))


# In[7]:


model.summary()


# In[8]:


datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary',
        shuffle='False')

validation_generator = datagen.flow_from_directory(
        valid_data_dir,
        target_size=(img_width, img_height),
        batch_size=10,
        class_mode='binary',
        shuffle='False')

print('complete')


# In[9]:


model.compile(optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])


# In[10]:


model.fit_generator(
        train_generator,
        steps_per_epoch=672,
        nb_epoch=5,
        validation_data=validation_generator,
        validation_steps=79)


# In[11]:


model.save_weights('Z:/Smoke_Classifier_weights/Vgg16_trial_2e.h5')


# In[12]:


model.load_weights('Z:/Smoke_Classifier_weights/Vgg16_trial_2e.h5')


# In[13]:


model.evaluate_generator(validation_generator,steps=79)


# # Using Validation Set to Predict (0 = No Smoke,1 = Smoke)

# In[14]:


model.predict_generator(validation_generator,steps=79)


# In[15]:


pred_FullSmoke_data_dir = 'Z:/SmokeDataset'

pred_generator = datagen.flow_from_directory(
        pred_FullSmoke_data_dir,
        target_size=(img_width, img_height),
        batch_size=10,
        class_mode=None,
        shuffle='False')


# # All the images fed here contain Smoke (0 = No Smoke, 1 = Smoke)

# In[16]:


model.predict_generator(pred_generator,steps=10,verbose=1)


# In[17]:


pred_NoSmoke_data_dir='Z:/KidPics'

pred_nsgenerator = datagen.flow_from_directory(
        pred_NoSmoke_data_dir,
        target_size=(img_width, img_height),
        batch_size=10,
        class_mode=None,
        shuffle='False')


# # All the images fed here does not contain smoke (0 = No Smoke, 1 = Smoke)

# In[18]:


model.predict_generator(pred_nsgenerator,steps=1,verbose=1)

