import numpy as np
import time
from matplotlib import pyplot as plt
from Models import Expert, IndexTreeBuilder

def experts_at_depth(root, all_experts, 
					 max_depth, cur_depth 
				 ):
	if root.check_leaf() or cur_depth == max_depth:
		all_experts.append((root.get_experts(), root.get_exemplars()))
		return

	else:
		for each_child in root.get_children():
			experts_at_depth(root = each_child, all_experts = all_experts, 
							 max_depth = max_depth, cur_depth = cur_depth +1)

def visualize_cls_separation(root, max_depth):
	"""
	We segregate experts based on their depth in the tree. 
	The deeper we go down, the more classes we expect to have.

	The root of the tree has depth of 0.
	Only experts at depth <= max_depth are identified
	"""
	experts_at_max_depth = []

	experts_at_depth(root = root, all_experts = experts_at_max_depth, 
					 max_depth = max_depth, cur_depth = 0)

	colors 	= ["r", "g", "b", "k", "m", "y", "c"]

	plt.xlabel("X") 
	plt.ylabel("Y")

	for indx, (each_experts, each_exemplars) in enumerate(experts_at_max_depth):
		cluster_indexes 	= [e.get_index() for e in each_experts]
		exemplar_indexes  	= [e.get_index() for e in each_exemplars]

		plt.scatter(*zip(*cluster_indexes), marker = "o", s = 10, c = colors[indx])
		plt.scatter(*zip(*exemplar_indexes), marker = "x", s = 100, c = colors[indx])

		for each_exemplar in each_exemplars:
			exemplar_tag    = each_exemplar.get_tags()
			exemplar_index  = each_exemplar.get_index()
			plt.annotate(f"{exemplar_tag}", (exemplar_index[0], exemplar_index[1]))

	plt.show()
	return

"""
We evaluate the correctness of building our indexed tree.
We first create a set of N experts and randomly assign each expert a 2-D index space coordinate.

We construct the indexed tree and plot a visual of how all experts at max_depth of the tree are clustered.
"""

if __name__ == "__main__":
	index_tree_builder 	= IndexTreeBuilder(	leaf_expert_count = 10, 
											k_clusters = 6,
											exemplar_count = 3
										)

	num_experts  	= 1000
	expert_index 	= np.random.uniform(low = 0, high = 1, size = (num_experts, 2)) 	# k x n, k being the number of experts and n the index dim
	experts 		= [Expert(trained_model = None, router = None) for _ in range(num_experts)]

	for indx, each_expert in enumerate(experts):
		each_expert.set_index(new_index = expert_index[indx])

	start 			= time.time()
	index_tree_root = index_tree_builder.build_index_tree(experts = experts)
	print(f"Time: {time.time() - start}")

	visualize_cls_separation(root = index_tree_root, max_depth = 1)