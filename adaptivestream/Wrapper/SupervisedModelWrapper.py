from abc import ABC
from abc import abstractmethod
from Wrapper.ModelWrapper import ModelWrapper

class SupervisedModelWrapper(ModelWrapper, ABC):
	def __init__(self, base_model, optimizer, *args, **kwargs):
		self.optimizer = optimizer
		super(SupervisedModelWrapper).__init__(base_model = base_model, *args, **kwargs)
		return

	@abstractmethod
	def train(self, input_X, output_Y, *args, **kwargs):
		pass