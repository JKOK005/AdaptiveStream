import sys
from Rules.Scaling.ScalingRules import ScalingRules

class TimeLimit(ScalingRules):
	def __init__(self, interval_sec: int):
		self.interval_sec = interval_sec
		return

	def check_scaling(self, buffer, *args, **kwargs) -> bool:
		"""
		Permits scaling only if batch data ingested is above max_ts - min_ts seconds
		"""
		[min_ts, max_ts] = buffer.get_batch_timestamps()
		return max_ts - min_ts >= self.interval_sec

	def reset(self):
		return