from abc import ABC
from abc import abstractmethod

class Buffer(ABC):
	def __init__(self, *args, **kwargs):
		pass

	@abstractmethod
	def get_data(self, *args, **kwargs):
		"""
		Returns stored data in the buffer
		"""
		pass

	@abstractmethod
	def add(self, data, *args, **kwargs):
		"""
		Adds new data to the buffer
		"""
		pass

	@abstractmethod
	def clear(self, *args, **kwargs):
		"""
		Clears buffer data
		"""
		pass