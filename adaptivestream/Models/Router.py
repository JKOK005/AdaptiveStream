from abc import ABC
from abc import abstractmethod
from Buffer.Buffer import Buffer

class Router(ABC):
	def __init__(self, base_model):
		pass 

	@abstractmethod
	def train(self, buffer: Buffer, 
					*args, **kwargs
			):
		pass

	@abstractmethod
	def permit_entry(self, *args, **kwargs):
		"""
		Evaluates if the input data is within the distribution of data router has been conditioned over.

		If true, permit entry to the expert.
		"""
		pass