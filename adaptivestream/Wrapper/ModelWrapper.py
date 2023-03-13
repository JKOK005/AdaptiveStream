from abc import ABC

class ModelWrapper(ABC):
	def __init__(self, base_model, *args, **kwargs):
		pass

	@abstractmethod
	def train(self, input_X, *args, **kwargs):
		"""
		User defines how the training process should work for choice of model.
		The end result should be a trained model. 
		"""
		pass

	@abstractmethod
	def infer(self, input_X, *args, **kwargs):
		"""
		User defines how the inference process should work for choice of model.
		The end result should be a return value representing model inference on the input data. 
		""" 
		pass