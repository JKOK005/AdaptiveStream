from Wrapper.ModelWrapper import ModelWrapper
from Routers.Router import Router

class Expert(object):
	self.trained_model 	= trained_model
	self.router 		= router

	def __init__(self, 	trained_model: ModelWrapper, 
						router: Router
				):
		pass

	def bypass(self, input_X):
		"""
		Bypass checks if input data is within the data distribution router is trained over. 
		If True, we allow this expert to perform inference over the data.
		"""
		return self.router.is_within_distribution(input_data)

	def infer(self, input_X):
		return self.trained_model.infer(input_X = input_X)