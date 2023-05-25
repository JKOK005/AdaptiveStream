from Models import ExpertEnsemble
from Rules.Checkpoint import CheckpointRules

class SaveOnStateChange(CheckpointRules):
	ensemble_tracked 	= None
	ensemble_prev_state = None

	def __init__(self, *args, **kwargs):
		return	

	def check_checkpoint(self, 	expert_ensemble: ExpertEnsemble, 
								*args, **kwargs
						):
		ensemble_current_state 		= expert_ensemble.get_state()["id"]
		state_switched 				= self.ensemble_prev_state != ensemble_current_state
		self.ensemble_prev_state 	= ensemble_current_state
		return state_switched