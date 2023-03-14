from abc import ABC

class Router(ABC):
	def __init__(self, base_model):
		pass 

	@abstractmethod
	def train(self, *args, **kwargs):
		pass

	@abstractmethod
	def is_within_distribution(self, *args, **kwargs):
		"""
		Evaluates if the input data is within the distribution of data router has been conditioned over
		"""
		pass