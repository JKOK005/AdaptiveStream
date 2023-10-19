from alibi_detect.cd import MMDDriftOnline
from Buffer.Buffer import Buffer
from Rules.Scaling import ScalingRules

class OnlineMMDDrift(ScalingRules):
	def __init__(self, 	min_trigger_count: int, 
						init_params: dict,
						safety_timestep: int
				):
		"""
		Scaling is done by first training the drift model detector over an initial set of data.
		We control the initial set of data using the 'min_trigger_count' parameter.

		The online mmd detector is trained using a starting dataset and an expected run time (ERT).
		Thresholds are configured such that the detector flags a drift on a non-drift sample window within a timestep close to ERT on average.
		For a sample window with drift, the timesteps taken is << ERT.

		To ensure that the model does not falsely flag a drift after ERT, we introduce a safety_timestep variable.
		If the timestep has crossed the safety_timestep threshold, we will reset the state of the model to t = 0.
		Hence, care has to be taken such that we do not set too low a safety_timestep value such that legitimate drift does not get detected.
		*Note: In deployment, we should set safety_timestep < ert.

		Once the detector is trained, we apply windowed sampling over new test instances to identify if data has drifted.
		For a clearer understanding of how drift is identified, please refer to [1]

		[1]: https://docs.seldon.io/projects/alibi-detect/en/stable/cd/methods/onlinemmddrift.html
		"""
		self.drift_model  		= None
		self.min_trigger_count 	= min_trigger_count
		self.init_params 		= init_params
		self.safety_timestep 	= safety_timestep
		self.ert_timesteps  	= 0
		return

	def check_scaling(self, buffer: Buffer, *args, **kwargs) -> bool:
		if buffer.get_count() >= self.min_trigger_count:
			if self.drift_model is None:
				# Train a new model only if one does not exist
				self.drift_model = MMDDriftOnline(	x_ref = buffer.get_data(), 
													backend = "tensorflow",
													**self.init_params
												)
			else:
				feats_latest 		= buffer.get_data_latest()
				batch_drift 		= [self.drift_model.predict(each_feats)["data"]["is_drift"] for each_feats in feats_latest]
				final_drift  		= max(batch_drift, key = batch_drift.count)
				self.ert_timesteps 	+= len(batch_drift)
			
				if self.ert_timesteps >= self.safety_timestep:
					self.ert_timesteps = 0
					self.drift_model.reset_state()
				
				return final_drift == 1
		return False

	def reset(self, *args, **kwargs):
		self.drift_model = None
		return