from abc import ABC
from abc import abstractmethod

class ScalingRules(ABC):
	@abstractmethod
	def check_scaling(self, buffer, *args, **kwargs) -> bool:
		pass