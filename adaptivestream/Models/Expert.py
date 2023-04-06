import numpy as np
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
		self.index  		= None
		return

	def permit_entry(self, input_X, *args, **kwargs):
		"""
		permit_entry checks if input data is within the data distribution router is trained over. 
		If True, we allow this expert to perform inference over the data.
		"""
		return self.router.permit_entry(input_X = input_X)

	def score(self, input_X, *args, **kwargs):
		"""
		Assigns a score to the new datapoint. The higher the score, the more likely the point is an outlier.
		"""
		return self.router.score(input_X = input_X)

	def infer(self, input_X, *args, **kwargs):
		return self.trained_model.infer(input_X = input_X)
	
	def evaluate(self, input_X, input_y, *args, **kwargs):
		return self.trained_model.evaluate(input_X = input_X, input_y = input_y)
	
	def loss(self, input_X, input_y, *args, **kwargs):
		return self.trained_model.loss(input_X = input_X, input_y = input_y)

	def get_index(self):
		return self.index

	def set_index(self, new_index: np.ndarray):
		self.index = new_index