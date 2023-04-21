from __future__ import annotations
from Math import ClusterTools
from Models import Expert, ExpertEnsemble
import numpy as np

class IndexedTreeNode(object):
	exemplar  	= None
	experts 	= []
	children  	= []
	is_leaf 	= False

	def __init__(self, 	experts: [Expert], 
						children: [IndexedTreeNode], 
						is_leaf: bool
				):
		self.experts 	= experts
		self.children 	= children
		self.is_leaf  	= is_leaf
		self.exemplar 	= self.compute_exemplar()

	def compute_exemplar(self):
		if len(self.experts) == 1:
			exemplar_indx = 0 
		else:
			expert_indexes 	= np.vstack([each_expert.get_index() for each_expert in self.experts])
			exemplar_indx  	= ClusterTools.exemplar_selection(indexes = expert_indexes)
		return self.experts[exemplar_indx]
	
	def get_experts(self) -> [Expert]:
		return self.experts

	def get_exemplar(self) -> Expert:
		return self.exemplar

	def get_children(self) -> [IndexedTreeNode]:
		return self.children

	def check_leaf(self) -> bool:
		return self.is_leaf

class IndexTreeBuilder(object):
	def __init__(self, 	leaf_expert_count: int, 
						k_clusters: int, 
						*args, **kwargs
				):
		"""
		params: leaf_expert_count 	: controls the number of experts per index tree node
		params: k_clusters 			: number of clusters to split the indexed tree
		"""
		self.leaf_expert_count 	= leaf_expert_count
		self.k_clusters 		= k_clusters
		return

	def build_index_tree(self, experts: [Expert]) -> IndexedTreeNode:
		if len(experts) <= self.leaf_expert_count:
			return 	IndexedTreeNode(
						experts 	= experts,
						children 	= [],
						is_leaf  	= True
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
					experts 	= experts,
					children 	= children,
					is_leaf  	= False
				)

class IndexedExpertEnsemble(ExpertEnsemble):
	indexed_tree = None

	def __init__(self, 	tree_builder: IndexTreeBuilder,
						*args, **kwargs
				):
		"""
		Class most shares similar features with ExpertEnsemble. 
		In addition, we build a K-means index over all frontier experts.
		This process allows search to be reduced from O(N) -> O(log N) to support high QPS for the application.

		params: tree_builder 	: Object to build index tree
		"""
		super(IndexedExpertEnsemble, self).__init__(*args, **kwargs)
		self.tree_builder 	= tree_builder
		return

	def _index_last(self):
		# TODO: Logic to index last expert after scaling
		pass

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
			self.indexed_tree 	= tree_builder.build_index_tree(experts = self.experts)

		return

	def infer(self, input_data):
		# TODO: Consult K means index tree for expert selection
		pass