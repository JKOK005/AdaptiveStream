from abc import ABC
from Wrapper.ModelWrapper import ModelWrapper
from Routers.Router import Router

class ScalingPolicy(ABC):
	self.model_template 	= None
	self.router_template 	= None
	self.buffer 			= None

	def __init__(self, 	model: ModelWrapper, 
						router: Router
				):
		self.model_template 	= model 
		self.router_template 	= router
		return

	def set_buffer(self, buffer):
		self.buffer = buffer
		return

	@abstractmethod
	def train_expert(self,  input_X, output_Y = None, 
							*args, **kwargs):
		pass