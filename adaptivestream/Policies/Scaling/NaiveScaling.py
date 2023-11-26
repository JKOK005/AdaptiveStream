import copy
import itertools
import time
from Models.Expert import Expert
from Models.Router.Router import Router
from Models.Wrapper.ModelWrapper import ModelWrapper
from Policies.Scaling.ScalingPolicy import ScalingPolicy

class NaiveScaling(ScalingPolicy):
	counter = itertools.count()

	def __init__(self, 	model: ModelWrapper, 
						router: Router,
						*args, **kwargs
				):
		super(NaiveScaling, self).__init__(model = model, router = router)
		return

	def train_expert(self, *args, **kwargs):
		expert_model 	= copy.deepcopy(self.model_template)
		expert_model.train(buffer = self.buffer)

		expert_router 	= copy.deepcopy(self.router_template)
		expert_router.train(buffer = self.buffer)

		trained_expert 	= Expert(trained_model = expert_model, router = expert_router)
		trained_expert.set_tags(tags = {"num" : next(self.counter)})
		return trained_expert

	def reset(self, *args, **kwargs):
		return