import datetime
from Rules.Scaling.ScalingRules import ScalingRules

class MaxDuration(ScalingRules):
	def __init__(self, max_duration: datetime.timedelta, *args, **kwargs):
		self.max_duration = duration
		return

	def check_scaling(self, buffer, *args, **kwargs) -> bool:
		"""
		Permits scaling only if buffer duration has exceeded max_duration.

		params: max_duration : maximum duration of data collection since the last buffer cleared timing
		"""
		current_time 	= datetime.datetime.now()
		last_cleared 	= buffer.get_last_cleared()
		return current_time - last_cleared >= self.max_duration

	def reset(self):
		return