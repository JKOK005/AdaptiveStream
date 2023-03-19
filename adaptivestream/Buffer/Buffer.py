from abc import ABC
from abc import abstractmethod
from datetime import datetime

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
	def get_count(self, *args, **kwargs):
		pass

	@abstractmethod
	def get_last_cleared(self, *args, **kwargs) -> datetime:
		"""
		Returns when the buffer was last cleared
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