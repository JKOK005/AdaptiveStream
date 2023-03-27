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

	def check_scaling(self, buffer: Buffer, *args, **kwargs) -> bool:
		"""
		Scaling is done by first training the drift model detector over an initial set of data.
		We control the initial set of data using the 'min_trigger_count' parameter.

		Once the detector is trained, we apply windowed sampling over new test instances to identify if data has drifted.
		For a clearer understanding of how drift is identified, please refer to [1]

		[1]: https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html
		"""
		if buffer.get_count() >= self.min_trigger_count:
			if self.drift_model is None:
				# Train a new model only if one does not exist
				self.drift_model = MMDDriftOnline(	x_ref = buffer.get_data(), 
													backend = "tensorflow",
													**self.init_params
												)
			else:
				feats_latest = buffer.get_data_latest()
				for each_feats in feats_latest:
					pred = self.drift_model.predict(each_feats)
				return pred['data']['is_drift'] == 1
		return False

	def reset(self, *args, **kwargs):
		self.drift_model = None
		return