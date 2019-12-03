from keras.models import Sequential
from keras import optimizers
from keras import losses
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, ZeroPadding2D, Activation
from keras.preprocessing.image import ImageDataGenerator
import h5py



#Training data directory
train_data_dir = "/dataset/train/"

#Test data directory
test_data_dir = "/dataset/test/"

#Model directory
model_dir = "/model/"



#Size of the image 
img_height,img_weight =  224,224

#Training batch size
train_batch_size = 16

#Test batch size
test_batch_size = 16

#Validation Split
validation_split = 0.20



#Creating a new image datagenerator
DataGen = ImageDataGenerator(rescale=1./255, validation_split = validation_split)

#Training data from directory
train_data = DataGen.flow_from_directory(
	train_data_dir,
	target_size = (img_height,img_weight),
	batch_size = train_batch_size,
	class_mode = 'binary',
	shuffle = False,
	subset = 'training')

#Validation data from directory
valid_data = DataGen.flow_from_directory(
	train_data_dir,
	target_size = (img_height,img_weight),
	batch_size = train_batch_size,
	class_mode = 'binary',
	shuffle = False,
	subset = 'validation')

#Testing data from directory
test_data = DataGen.flow_from_directory(
	test_data_dir,
	target_size = (img_height,img_weight),
	batch_size = test_batch_size,
	class_mode = None.
	shuffle = False)



def model:

	#Define your classifer here
	classifer = Sequential()

	return classifier



#Get the defined model
classifier = model()

#Model optimizer
optimizer = optimizers.Adam(learning_rate=0.001)

#Model loss
loss = 'binary_crossentropy'

#Model metrics
metrics = ['accuracy']



classifier.compile(optimizer = optimizer, loss = loss, metrics = metrics)

classifier.save(model_dir + 'classifier_model.h5')



steps_per_epoch = len(train_data)/train_batch_size

valid_steps = len(valid_steps)/train_batch_size

epochs = 500



classifier.fit_generator(
	train_data,
	steps_per_epoch = steps_per_epoch,
	epochs = epochs,
	verbose = 2,
	validation_data = valid_data,
	validation_steps = valid_steps)



classifier.save(model_dir + 'classifier_model.h5')