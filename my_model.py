import Keras.layers
import Keras.models
import Keras.optimizers
import Keras.losses
import configs as configs

class Model:

	self.model = Sequential()

	self.model_dir = configs.model_dir

	self.img_height,self.img_weight =  configs.img_height,configs.img_weight

	#Model optimizer
	self.optimizer = optimizers.Adam(learning_rate=0.001)

	#Model loss
	self.loss = configs.model_loss

	#Model metrics
	self.metrics = configs.model_metrics

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

		self.model.save(self.model_dir + 'model.h5')

	def __CompileModel(self):

		self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)

if __name__ == ('__main__'):

	model_maker = Model()


