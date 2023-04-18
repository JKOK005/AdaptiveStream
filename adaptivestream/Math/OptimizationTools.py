import numpy as np
import tensorflow as tf
from Models import Expert
from tensorflow.keras.losses import KLDivergence
from typing import Dict

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
	def optimize(expert_index: np.array, 
				 target_dist: np.array,
				 epochs: int, 
				 early_stopping_tol: float = 1e-4,
				 optim_params
			) -> np.array:
		"""
		Given 
		1) k x n expert_index matrice, k being the number of experts and n the dimension of the index space
		2) target_distribution in R^k space

		Return a R^n array X* that minimizes the discrepancy between 
		1) The softmax distribution of distance for each expert to X* 
		2) The target distribution

		We will use the KL-divergence to compute distribution differences
		"""
		assigned_index 	= tf.Variable(tf.random.uniform(shape = (expert_index.shape[-1],)), trainable = True)
		optimizer 		= tf.keras.optimizers.Adam(**optim_params)
		loss 			= KLDivergence()

		for _ in range(epochs):
			with tf.GradientTape() as tape:
				dist  		= tf.math.reduce_sum((expert_index - assigned_index) ** 2. axis = 1)
				pred_dist 	= tf.nn.softmax(dist)
				cost 		= loss(target_dist, pred_dist)
			
			grad = tape.gradient(loss, assigned_index)
			optimizer.apply_gradients([(grad, assigned_index)])

			if tf.math.reduce_sum(grad ** 2. axis = 0) <= early_stopping_tol:
				break
		return assigned_index.numpy()
