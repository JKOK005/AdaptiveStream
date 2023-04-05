import sys
from Rules.Scaling.ScalingRules import ScalingRules

class BufferSizeLimit(ScalingRules):
	def __init__(self, 	min_size: int = 0,
						max_size: int = sys.maxsize
				):

		self.min_size = min_size
		self.max_size = max_size
		return

	def check_scaling(self, buffer, *args, **kwargs) -> bool:
		"""
		Permits scaling only if buffer size is within min_size - max_size bounds
		"""
		count = buffer.get_count()
		return count >= self.min_size and count <= self.max_size

	def reset(self):
		return