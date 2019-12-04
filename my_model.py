import Keras.layers
import Keras.models
import Keras.optimizers
import Keras.losses

class Model:

	self.model = Sequential()

	self.model_dir = '/model/'

	self.img_height,self.img_weight =  224,224

	#Model optimizer
	self.optimizer = optimizers.Adam(learning_rate=0.001)

	#Model loss
	self.loss = 'binary_crossentropy'

	#Model metrics
	self.metrics = ['accuracy']

	def __init__(self):

		self.model = Sequential()

		self.__CreateModel()
		self.__CompileModel()
		self.__SaveModel(self.model_dir)
	
	def __getitem__(self):

		return self.model

	def __CreateModel(self):

		#Define your model here

	def __SaveModel(self,model_dir):

		self.model.save(model_dir + 'model.h5')

	def __CompileModel(self):

		self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

if __name__ == ('__main__'):

	model_maker = Model()


