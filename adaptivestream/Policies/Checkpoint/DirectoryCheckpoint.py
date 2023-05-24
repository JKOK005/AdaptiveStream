import logging
import os
import pickle
from Policies.Checkpoint import CheckpointPolicy
from pathlib import Path

class DirectoryCheckpoint(CheckpointPolicy):
	def __init__(self, 	save_path: str,
						create_if_exists: bool = True,
						*args, **kwargs):
		
		self.save_path 	= save_path
		self.logger  	= logging.getLogger("DirectoryCheckpoint")
		
		Path(save_path).mkdir(parents = create_if_exists, exist_ok = True)
		logging.info(f"Logging model to: {self.save_path}")
		return

	def save(self, 	expert_emsemble: ExpertEnsemble, 
					*args, **kwargs
			):
		pass