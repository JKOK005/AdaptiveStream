from __future__ import annotations
from Math import ClusterTools, OptimizationTools
from Models import Expert, ExpertEnsemble
import numpy as np
import tensorflow as tf

class IndexedTreeNode(object):
	experts 	= []
	exemplars  	= []
	children  	= []
	is_leaf 	= False

	def __init__(self, 	experts: [Expert], 
						children: [IndexedTreeNode],
						exemplar_count: int, 
						is_leaf: bool
				):
		self.experts 	= experts
		self.children 	= children
		self.is_leaf  	= is_leaf
		self.exemplars 	= self.compute_exemplars(max_count = exemplar_count) if len(experts) > 0 else None

	def compute_exemplars(self, max_count: int):
		if len(self.experts) <= max_count:
			exemplars  		= self.experts
		else:
			expert_indexes 	= np.vstack([each_expert.get_index() for each_expert in self.experts])
			exemplar_indx  	= ClusterTools.exemplar_selection(indexes = expert_indexes, exemplar_count = max_count)
			exemplars  		= [self.experts[i] for i in exemplar_indx]
		return exemplars
	
	def get_experts(self) -> [Expert]:
		return self.experts

	def get_exemplars(self) -> [Expert]:
		return self.exemplars

	def get_children(self) -> [IndexedTreeNode]:
		return self.children

	def check_leaf(self) -> bool:
		return self.is_leaf

class IndexTreeBuilder(object):
	def __init__(self, 	leaf_expert_count: int, 
						k_clusters: int, 
						exemplar_count: int, 
						*args, **kwargs
				):
		"""
		params: leaf_expert_count 	: controls the number of experts per index tree node
		params: k_clusters 			: number of clusters to split the indexed tree
		params: exemplar_count 		: K exemplars used to represent the cluster
		"""
		self.leaf_expert_count 	= leaf_expert_count
		self.k_clusters 		= k_clusters
		self.exemplar_count 	= exemplar_count
		return

	def build_index_tree(self, experts: [Expert]) -> IndexedTreeNode:
		if len(experts) <= self.leaf_expert_count:
			return 	IndexedTreeNode(
						exemplar_count 	= self.exemplar_count,
						experts 		= experts,
						children 		= [],
						is_leaf  		= True
					)

		expert_indexes 	= np.vstack([each_expert.get_index() for each_expert in experts])
		expert_class  	= ClusterTools.k_means_cluster(indexes = expert_indexes, clusters = self.k_clusters, n_init = 10)
		children 		= []

		for each_cls in range(self.k_clusters):
			individual_cls_indx  	= np.where(expert_class == each_cls)[0]
			individual_cls_experts 	= [experts[indx] for indx in individual_cls_indx]
			child_node  			= self.build_index_tree(experts = individual_cls_experts)
			children.append(child_node)

		return 	IndexedTreeNode(
					exemplar_count 	= self.exemplar_count,
					experts 		= experts,
					children 		= children,
					is_leaf  		= False
				)

class IndexedExpertEnsemble(ExpertEnsemble):
	indexed_tree 	= None
	tree_builder 	= None
	index_dim 		= None

	def __init__(self, 	tree_builder: IndexTreeBuilder,
						index_dim: int, 
						*args, **kwargs
				):
		"""
		Class most shares similar features with ExpertEnsemble. 
		In addition, we build a K-means index over all frontier experts.
		This process allows search to be reduced from O(N) -> O(log N) to support high QPS for the application.

		params: tree_builder 	: Object to build index tree
		params: index_dim 		: R^(dim) space for each expert index
		"""
		super(IndexedExpertEnsemble, self).__init__(*args, **kwargs)
		self.tree_builder 	= tree_builder
		self.index_dim  	= index_dim
		return

	def _index_last(self):
		if len(self.experts) > 0:
			latest_expert = self.experts[-1]

			if len(self.experts) <= 2:
				assigned_index 				= np.random.uniform(low = 0, high = 1, size = (self.index_dim))

			else:
				latest_batch_data  			= self.buffer.get_data()
				historical_experts 			= self.experts[:-1]
				historical_experts_index 	= np.vstack([each_expert.get_index() for each_expert in historical_experts])

				latest_batch_score  = tf.stack([
										OptimizationTools.loss_dist(
											expert_set = historical_experts, 
											input_X = tf.expand_dims(each_data, axis = 0)
										) for each_data in latest_batch_data
									], axis = 0)
				
				average_score  		= tf.math.reduce_mean(latest_batch_score, axis = 0)
				target_dist  		= tf.nn.softmax(average_score, axis = 0)

				assigned_index 		= OptimizationTools.optimize(
											expert_index = historical_experts_index,
											target_dist = target_dist,
											epochs = 10000,
											early_stopping_tol = 1e-6,
											l2_ratio = 0.01,
											optim_params = {"learning_rate" : 0.05},
										)

			print(f"Assigned index: {assigned_index}")
			latest_expert.set_index(new_index = assigned_index)
		return

	def ingest(self, batch_input):
		self.buffer.add(batch_input = batch_input)

		is_compact 	= self._check_to_compact()
		is_scale  	= self._check_to_scale()
			
		if is_compact:
			(new_fallback_expert, new_experts) 	= self.compaction_policy.compact(
													expert_chain = self.experts, 
													prev_fallback_expert = self.fallback_expert, 
												)

			self.fallback_expert = new_fallback_expert
			self.experts = new_experts

		if is_scale:
			self.scale_experts()
			self._index_last()
			self._reset_scale()

		if is_compact or is_scale:
			# Rebuild K means index tree
			self.indexed_tree 	= self.tree_builder.build_index_tree(experts = self.experts)
		return

	def infer(self, input_data):
		# TODO: Evaluate on fall back expert logic

		def leaf_selection(root: IndexedTreeNode) -> IndexedTreeNode:
			if root.check_leaf():
				return root

			children_nodes 		= root.get_children()
			children_exemplars 	= [each_node.get_exemplars() for each_node in children_nodes]
			scores  			= np.array([each_exemplar.score(input_X = input_data) for each_exemplar in children_exemplars])
			best_children  		= scores.argmin()
			return leaf_selection(root = children_nodes[best_children])
		
		leaf_node 		= leaf_selection(root = self.indexed_tree)
		leaf_experts 	= leaf_node.get_experts()
		preds 			= [each_expert.infer(input_data) for each_expert in leaf_experts]
		return tf.math.reduce_mean(preds, axis = 0)