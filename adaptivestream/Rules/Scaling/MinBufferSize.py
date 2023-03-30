from Rules.Scaling.ScalingRules import ScalingRules

class MinBufferSize(ScalingRules):
	def __init__(self, 	min_size: int):

		self.min_size = min_size
		return

	def check_scaling(self, buffer, *args, **kwargs) -> bool:
		"""
		Permits scaling only if buffer size has exceeded min_size.
		"""
		count = buffer.get_count()
		return count >= self.min_size

	def reset(self):
		return