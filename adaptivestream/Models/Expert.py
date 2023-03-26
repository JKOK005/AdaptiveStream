from Models.Router.Router import Router
from Models.Wrapper.ModelWrapper import ModelWrapper

class Expert(object):
	trained_model 	= None
	router 			= None

	def __init__(self, 	trained_model: ModelWrapper, 
						router: Router,
						*args, **kwargs
				):
		self.trained_model 	= trained_model
		self.router 		= router
		return

	def permit_entry(self, input_X, *args, **kwargs):
		"""
		permit_entry checks if input data is within the data distribution router is trained over. 
		If True, we allow this expert to perform inference over the data.
		"""
		return self.router.permit_entry(input_X = input_X)

	def infer(self, input_X, *args, **kwargs):
		return self.trained_model.infer(input_X = input_X)