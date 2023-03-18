from abc import ABC
from abc import abstractmethod

class CompactionPolicy(ABC):
	buffer 	= None

	def set_buffer(self, buffer):
		self.buffer = buffer
		return

	@abstractmethod
	def compact(self, experts, 
					  fallback_expert,
					  *args, **kwargs
				):
		pass