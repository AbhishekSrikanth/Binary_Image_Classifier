from keras.models
from keras.preprocessing.image import ImageDataGenerator
import h5py
import configs as configs


#Training data directory
train_data_dir = configs.train_data_dir

#Test data directory
test_data_dir = configs.test_data_dir

#Model directory
model_dir = configs.model_dir

classifier = models.load_model(model_dir + 'model.h5')

#Size of the image 
img_height,img_weight =  configs.img_height,configs.img_weight

#Training batch size
train_batch_size = configs.train_batch_size

#Test batch size
test_batch_size = configs.test_batch_size

#Validation Split
validation_split = configs.validation_split



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


#Number Steps * Batch Size = Total dataset size
steps_per_epoch = len(train_data)/train_batch_size

valid_steps = len(valid_steps)/train_batch_size

epochs = configs.epochs



classifier.fit_generator(
	train_data,
	steps_per_epoch = steps_per_epoch,
	epochs = epochs,
	verbose = 2,
	validation_data = valid_data,
	validation_steps = valid_steps)



classifier.save(model_dir + 'model.h5')