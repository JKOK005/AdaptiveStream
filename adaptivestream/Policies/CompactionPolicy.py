from abc import ABC

class CompactionPolicy(ABC):
	self.buffer 	= None

	def set_buffer(self, buffer):
		self.buffer = buffer
		return

	@abstractmethod
	def trigger_compaction(self, expert_chain, 
								 *args, **kwargs
						 ):
		pass

	@abstractmethod
	def compact(self, expert_chain, 
					  prev_fallback_expert,
					  *args, **kwargs
				):
		pass