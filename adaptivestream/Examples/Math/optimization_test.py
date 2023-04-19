import numpy as np
import time
from Math import OptimizationTools
from matplotlib import pyplot as plt
from scipy.special import softmax

"""
Sanity check for index assignment logic.
We generate K number of experts in 2-D space with random coordinates.
We select a target expert and set the objective to be the distance between all points and that expert's index.
We expect the assigned index to be close in 2-D space to the index of the target expert.
"""

if __name__ == "__main__":
	num_experts  	= 1000
	selected_expert = 10

	expert_index 	= np.random.uniform(low = 0, high = 1, size = (num_experts, 2)) 	# k x n, k being the number of experts and n the index dim
	target_dist     = np.sum((expert_index - expert_index[selected_expert]) ** 2, axis = 1)
	target_dist  	= softmax(target_dist)
	
	start = time.time()

	assigned_index  = OptimizationTools.optimize(expert_index = expert_index,
												 target_dist = target_dist,
												 epochs = 10000,
												 early_stopping_tol = 1e-6,
												 l2_ratio = 0.01,
												 optim_params = {"learning_rate" : 0.05},
												)

	print(f"Time: {time.time() - start}")

	plt.xlabel("X") 
	plt.ylabel("Y") 

	plt.scatter(expert_index[:,0], expert_index[:,1], 
				marker = "o", c = "b")

	for i in range(expert_index.shape[0]):
		plt.annotate(f"{i}", (expert_index[i,0], expert_index[i,1]))

	plt.scatter(assigned_index[0], assigned_index[1], marker = "x", c = "r")
	plt.scatter(expert_index[selected_expert][0], expert_index[selected_expert][1], marker = "o", c = "r")
	plt.show()