import tensorflow as tf
from abc import ABC
from abc import abstractmethod
from Buffer.Buffer import Buffer

class LabelledBuffer(Buffer):
	def __init__(self, *args, **kwargs):
		super(LabelledBuffer, self).__init__(*args, **kwargs)
		return

	@abstractmethod
	def get_label(self, *args, **kwargs) -> tf.Tensor:
		"""
		Returns labels for each data in buffer
		"""
		pass