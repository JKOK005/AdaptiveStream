from abc import ABC
from abc import abstractmethod
from Buffer.Buffer import Buffer
from Models.Wrapper.ModelWrapper import ModelWrapper
from Models.Router.Router import Router

class ScalingPolicy(ABC):
	model_template 	= None
	router_template = None
	buffer 			= None

	def __init__(self, 	model: ModelWrapper, 
						router: Router,
						*args, **kwargs
				):
		self.model_template 	= model 
		self.router_template 	= router
		return

	def set_buffer(self, buffer: Buffer):
		self.buffer = buffer
		return

	@abstractmethod
	def train_expert(self, *args, **kwargs):
		pass

	@abstractmethod
	def reset(self, *args, **kwargs):
		pass