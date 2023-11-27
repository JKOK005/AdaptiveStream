import tensorflow as tf
from Buffer.Buffer import Buffer
from Models.Router.Router import Router

class DummyRouter(Router):
	def __init__(self):
		return

	def train(self, buffer: Buffer,
					*args, **kwargs
			):
		return

	def permit_entry(self, 	input_X: tf.Tensor, 
					  		*args, **kwargs
	  				) -> bool:
		return True

	def score(self, input_X: tf.Tensor,
					*args, **kwargs
			) -> float:
		return 1

	def prob(self, 	input_X: tf.Tensor,
					*args, **kwargs
			) -> float:
		return 1