#model directory
model_dir = '/model/'

#Image size in pixels
img_height,img_weight =  224,224

#Model loss
model_loss = 'binary_crossentropy'

#Model metrics
model_metrics = ['accuracy']

#Training data directory
train_data_dir = "/dataset/train/"

#Test data directory
test_data_dir = "/dataset/test/"

#Training batch size
train_batch_size = 16

#Test batch size
test_batch_size = 16

#Validation Split
validation_split = 0.20

#Number of epochs
epochs = 500

Label1 = 'ClassA'

Label2 = 'ClassB'