import numpy as np
import time
from Models import Expert, IndexTreeBuilder

if __name__ == "__main__":
	index_tree_builder = IndexTreeBuilder(	leaf_expert_count = 3, 
											k_clusters = 3)

	num_experts  	= 1000
	expert_index 	= np.random.uniform(low = 0, high = 1, size = (num_experts, 2)) 	# k x n, k being the number of experts and n the index dim
	experts 		= [Expert(trained_model = None, router = None) for _ in range(num_experts)]

	for indx, each_expert in enumerate(experts):
		each_expert.set_index(new_index = expert_index[indx])

	index_tree_root = index_tree_builder.build_index_tree(experts = experts)