from alibi_detect.cd import MMDDriftOnline
from Buffer.Buffer import Buffer
from Rules.Scaling import ScalingRules

class OnlineMMDDrift(ScalingRules):
	def __init__(self, 	min_trigger_count: int, 
						init_params: dict, 
				)
	
		self.drift_model  		= None
		self.min_trigger_count 	= min_trigger_count
		self.init_params 		= init_params
		return

	def check_scaling(self, buffer: Buffer, *args, **kwargs):
		pass