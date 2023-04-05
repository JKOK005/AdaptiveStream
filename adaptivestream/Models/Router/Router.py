from abc import ABC
from abc import abstractmethod
from Buffer.Buffer import Buffer

class Router(ABC):
	@abstractmethod
	def train(self, buffer: Buffer, 
					*args, **kwargs
			):
		pass

	@abstractmethod
	def permit_entry(self, *args, **kwargs) -> bool:
		"""
		Evaluates if the input data is within the distribution of data router has been conditioned over.

		If true, permit entry to the expert.
		"""
		pass

	@abstractmethod
	def score(self, *args, **kwargs) -> float:
		"""
		Score measures the degree of which the data is considered as an outlier to the trained distribution.

		The higher the score, the more the data is regarded as an outlier.
		ATTN: It is important to follow this convention when building future router classes.
		"""
		pass