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
		super(NaiveScaling).__init__(model = model, router = router)
		return

	def train_expert(self, input_X, output_Y = None):
		expert_model 	= copy.deepcopy(self.model_template)
		expert_model.train(input_X = input_X, output_Y = output_Y)

		expert_router 	= copy.deepcopy(self.router_template)
		expert_router.train(input_X = input_X)

		trained_expert 	= Expert(trained_model = expert_model, router = expert_router)
		return trained_expert