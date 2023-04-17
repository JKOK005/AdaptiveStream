import numpy as np
from Models import Expert
from typing import Dict
from scipy.optimize import minimize

class OptimizationTools(object):
	@staticmethod
	def loss_dist(	expert_set: [Expert], 
					input_X: np.array
				) -> np.array:
		"""
		Generates loss distribution of single expert relative to all other experts in the set, given a single instance of input_X.

		We use the score function in each router as a measure the loss between an expert and the test data point.
		"""
		loss = [expert.score(input_X = input_X) for expert in expert_set]
		return np.array(loss)

	@staticmethod
	def assign_index(expert_index: np.array, 
					 target_dist: np.array,
					 optim_params
				 ):
		pass