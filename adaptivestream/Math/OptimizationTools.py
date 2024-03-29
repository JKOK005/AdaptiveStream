import numpy as np
import tensorflow as tf
from Models import Expert
from tensorflow.keras.losses import KLDivergence
from typing import Dict

class OptimizationTools(object):
	@staticmethod
	def loss_dist(	expert_set: [Expert], 
					input_X: tf.Tensor
				) -> tf.Tensor:
		"""
		Generates loss distribution of single expert relative to all other experts in the set, given a single instance of input_X.

		We use the score function in each router as a measure the loss between an expert and the test data point.

		params: expert_set 	: Set of experts to compute router outlier score over.
		params: input_X 	: Singular input tensor to evaluate expert score over.
		"""
		loss = [expert.score(input_X = input_X) for expert in expert_set]
		return tf.convert_to_tensor(loss)

	@staticmethod
	def optimize(expert_index: np.array, 
				 target_dist: tf.Tensor,
				 epochs: int, 
				 optim_params,
				 early_stopping_tol: float = 1e-4,
				 l2_ratio: float = 0.1
			) -> np.array:
		"""
		Return a R^n array X* that minimizes the discrepancy between 
		1) The softmax distribution of distance for each expert to X* 
		2) The target distribution

		We will use the KL-divergence to compute distribution differences

		params: expert_index 		: k x n expert_index matrice, k being the number of experts and n the dimension of the index space.
		params: target_dist 		: target_distribution in R^k space.
		params: epochs 				: Number of epoches to run optimization.
		params: optim_params 		: ADAM optimizer params
		params: early_stopping_tol 	: Terminates optimization when | gradient | <= early_stopping_tol
		params: l2_ratio 			: l2 normalization weightage
		"""
		assigned_index 	= tf.Variable(
							tf.random.uniform(	shape = (expert_index.shape[-1],), 
												minval = 0, 
												maxval = 1
											), 
							trainable = True
						)

		optimizer 		= tf.keras.optimizers.Adam(**optim_params)
		loss 			= KLDivergence()

		for _ in range(epochs):
			with tf.GradientTape() as tape:
				dist  		= tf.math.reduce_sum((expert_index - assigned_index) ** 2, axis = 1)
				pred_dist 	= tf.nn.softmax(dist)
				cost 		= (1 - l2_ratio) * loss(target_dist, pred_dist) + l2_ratio * tf.math.reduce_sum(assigned_index ** 2, axis = 0)

			grad = tape.gradient(cost, assigned_index)
			optimizer.apply_gradients([(grad, assigned_index)])

			if tf.math.reduce_sum(grad ** 2, axis = 0) <= early_stopping_tol:
				break

		return assigned_index.numpy()
