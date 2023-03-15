from abc import ABC
from abc import abstractmethod

class CompactionRules(ABC):
	@abstractmethod
	def check_compaction(self, experts, *args, **kwargs):
		pass