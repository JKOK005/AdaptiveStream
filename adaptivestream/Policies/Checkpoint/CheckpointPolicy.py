from abc import ABC
from abc import abstractmethod
from Models import ExpertEnsemble

class CheckpointPolicy(ABC):
	def __init__(self, *args, **kwargs):
		return

	@abstractmethod
	def save(self, 	expert_emsemble: ExpertEnsemble, 
					*args, **kwargs
			):
		pass