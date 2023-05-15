import tensorflow as tf
from abc import ABC
from abc import abstractmethod
from datetime import datetime

class Buffer(ABC):
	batch_timestamp_range = [-1, -1]

	def __init__(self, *args, **kwargs):
		pass

	@abstractmethod
	def get_batch_timestamps(self, *args, **kwargs):
		"""
		Returns range of batch timestamps
		"""
		pass

	@abstractmethod
	def get_data(self, *args, **kwargs) -> tf.Tensor:
		"""
		Returns stored data in the buffer
		"""
		pass

	@abstractmethod
	def get_data_latest(self, *args, **kwargs) -> tf.Tensor:
		"""
		Returns the most recent batch of added data
		"""
		pass

	@abstractmethod
	def get_count(self, *args, **kwargs) -> int:
		pass

	@abstractmethod
	def get_last_cleared(self, *args, **kwargs) -> datetime:
		"""
		Returns when the buffer was last cleared
		"""
		pass

	@abstractmethod
	def add(self, 	batch_input: (tf.Tensor), 
					batch_timestamp: int,
					*args, **kwargs
			):
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