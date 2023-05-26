from Models import ExpertEnsemble
from abc import ABC
from abc import abstractmethod

class CheckpointRules(ABC):
	@abstractmethod
	def check_checkpoint(self, 	expert_ensemble: ExpertEnsemble, 
								*args, **kwargs
						):
		pass