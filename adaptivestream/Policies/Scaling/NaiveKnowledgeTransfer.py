import copy
from Models.Expert import Expert
from Models.Router.Router import Router
from Models.Wrapper.ModelWrapper import ModelWrapper
from Policies.Scaling.ScalingPolicy import ScalingPolicy

class NaiveKnowledgeTransfer(ScalingPolicy):
	prior_expert = None

	def __init__(self, 	model: ModelWrapper, 
						router: Router,
						*args, **kwargs
				):
		super(NaiveKnowledgeTransfer, self).__init__(model = model, router = router)
		return

	def train_expert(self, *args, **kwargs):
		expert_model 		= 	copy.deepcopy(self.model_template) if self.prior_expert is None \
								else copy.deepcopy(self.prior_expert)

		expert_model.train(buffer = self.buffer)
		self.prior_expert 	= expert_model

		expert_router 	= copy.deepcopy(self.router_template)
		expert_router.train(buffer = self.buffer)

		trained_expert 	= Expert(trained_model = expert_model, router = expert_router)
		return trained_expert

	def reset(self, *args, **kwargs):
		return