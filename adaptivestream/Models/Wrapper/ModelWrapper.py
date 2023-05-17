from abc import ABC
from abc import abstractmethod
from Buffer.Buffer import Buffer

class ModelWrapper(ABC):
	def __init__(self, *args, **kwargs):
		pass

	@abstractmethod
	def train(self, buffer: Buffer, 
					*args, **kwargs
			):
		"""
		User defines how the training process should work for choice of model.
		The end result should be a trained model. 
		"""
		pass

	@abstractmethod
	def infer(self, input_X, 
					*args, **kwargs
			):
		"""
		User defines how the inference process should work for choice of model.
		The end result should be a return value representing model inference on the input data. 
		""" 
		pass

	def loss(self, 	input_X, 
					ground_truth, 
					*args, **kwargs
			):
		"""
		Scores the prediction of a model over the given ground truth.
		"""
		pass