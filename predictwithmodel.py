import tensorflow as tf
from tf.keras import models
import configs as configs


class PredictWithModel:

    def __init__(self):
        self.model_dir = configs.model_dir
        self.model = models.load_model(self.model_dir + '/model.h5')

    def getPredictions(self, image):
        predictions = self.model.predict(image)
        return predictions
