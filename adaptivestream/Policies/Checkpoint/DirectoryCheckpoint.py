import logging
import os
import pickle
import time
from Models import ExpertEnsemble
from Policies.Checkpoint import CheckpointPolicy
from pathlib import Path

class DirectoryCheckpoint(CheckpointPolicy):
	def __init__(self, 	save_path: str,
						create_if_exists: bool = True,
						*args, **kwargs):
		
		self.save_path 	= save_path
		self.logger  	= logging.getLogger("DirectoryCheckpoint")
		
		Path(save_path).mkdir(parents = create_if_exists, exist_ok = True)
		logging.info(f"Base model dir is {self.save_path}")
		return

	def save(self, 	expert_emsemble: ExpertEnsemble, 
					log_state: bool = False,
					*args, **kwargs
			):
		
		current_time_round_up 	= int(time.time())
		model_save_path 		= os.path.join(self.save_path, str(current_time_round_up))
		model_save_file  		= os.path.join(model_save_path, "model.pickle")

		Path(model_save_path).mkdir(parents = False, exist_ok = False)

		with open(model_save_file, 'wb') as f:
			pickle.dump(expert_emsemble, f)
		
		if log_state:
			state  = expert_emsemble.get_state()
			timestamp_path 	= os.path.join(model_save_path, "state.txt")
			
			with open(timestamp_path, "w") as f:
				f.write(str(state))

		logging.info(f"Save model to: {model_save_path}")
		return