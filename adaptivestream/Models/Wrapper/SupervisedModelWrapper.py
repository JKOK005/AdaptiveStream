import tensorflow as tf
import logging
from Buffer.Buffer import Buffer
from Models.Wrapper.ModelWrapper import ModelWrapper

class SupervisedModelWrapper(ModelWrapper):
	def __init__(self, 	base_model: tf.keras.Model, 
						optimizer: tf.keras.optimizers, 
						loss: tf.keras.losses,
						training_params: dict,
						*args, **kwargs
				):
		super(SupervisedModelWrapper, self).__init__()
		self.loss 				= loss
		self.model 				= base_model
		self.optimizer 			= optimizer
		self.training_params 	= training_params
		self.logger  			= logging.getLogger("SupervisedModelWrapper")
		return

	def train(self, buffer: Buffer, *args, **kwargs):
		buffer_feat 	= buffer.get_data()
		buffer_label 	= buffer.get_label()

		self.model.compile(optimizer = self.optimizer, loss = self.loss)
		self.model.fit(buffer_feat, buffer_label, **self.training_params)
		return

	def infer(self, input_X, *args, **kwargs):
		pass