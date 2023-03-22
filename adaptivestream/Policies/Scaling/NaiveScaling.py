import copy
from Models.Expert import Expert
from Models.Router import Router
from Policies.Scaling.ScalingPolicy import ScalingPolicy
from Wrapper.ModelWrapper import ModelWrapper

class NaiveScaling(ScalingPolicy):
	def __init__(self, 	model: ModelWrapper, 
						router: Router,
						*args, **kwargs
				):
		super(NaiveScaling, self).__init__(model = model, router = router)
		return

	def train_expert(self):
		expert_model 	= copy.deepcopy(self.model_template)
		expert_model.train(buffer = self.buffer)

		expert_router 	= copy.deepcopy(self.router_template)
		expert_router.train(buffer = self.buffer)

		trained_expert 	= Expert(trained_model = expert_model, router = expert_router)
		return trained_expert