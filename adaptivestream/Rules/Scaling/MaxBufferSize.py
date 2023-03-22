from Rules.Scaling.ScalingRules import ScalingRules

class MaxBufferSize(ScalingRules):
	def __init__(self, max_size: int):
		self.max_size = max_size
		return

	def check_scaling(self, buffer, *args, **kwargs) -> bool:
		"""
		Permits scaling only if buffer size has exceeded max_size.

		params: max_size : maximum size of buffer before call to scale
		"""
		count = buffer.get_counts()
		return count >= self.max_size