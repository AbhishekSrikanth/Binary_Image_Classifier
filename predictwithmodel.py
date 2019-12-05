import keras.models
import configs as configs

class PredictWithModel:

	def __init__(self):

		self.model_dir = configs.model_dir
		self.model = models.load_model(model_dir + '/model.h5')

	def getPredictions(self,image):

		predictions = self.model.predict(image)
		return predictions