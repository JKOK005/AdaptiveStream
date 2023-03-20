from abc import ABC
from abc import abstractmethod
from Buffer.Buffer import Buffer

class LabelledBuffer(Buffer):
	def __init__(self, *args, **kwargs):
		super(LabelledBuffer).__init__(*args, **kwargs)
		return

	@abstractmethod
	def get_label(self, *args, **kwargs):
		"""
		Returns labels for each data in buffer
		"""
		pass