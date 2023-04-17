from Math import CusterTools
from Models.ExpertEnsemble import ExpertEnsemble
from operator import itemgetter

class IndexedTreeNode(object):
	exemplar  	= None
	experts 	= []
	children  	= []
	is_leaf 	= False

	def __init__(self, 	experts, 
						children, 
						is_leaf
				):
		self.experts 	= experts
		self.children 	= children
		self.is_leaf  	= is_leaf
		self.exemplar 	= self.compute_exemplar()

	def compute_exemplar(self):
		expert_indexes 	= np.concatenate([each_expert.get_index() for each_expert in self.experts], axis = 0)
		exemplar_indx  	= ClusterTools.exemplar_selection(indexes = expert_indexes)
		return self.experts[exemplar_indx]
	
	def get_experts(self):
		return self.experts

	def get_exemplar(self):
		return self.exemplar

	def get_children(self):
		return self.children

	def is_leaf(self):
		return self.is_leaf


class IndexedExpertEnsemble(ExpertEnsemble):
	indexed_tree = None

	def __init__(self, 	leaf_expert_count: int,
						k_clusters: int,
						*args, **kwargs
				):
		"""
		Class most shares similar features with ExpertEnsemble. 
		In addition, we build a K-means index over all frontier experts.
		This process allows search to be reduced from O(N) -> O(log N) to support high QPS for the application.

		params: leaf_expert_count : controls the number of experts per index tree node
		"""
		super(IndexedExpertEnsemble, self).__init__(*args, **kwargs)
		self.leaf_expert_count 	= leaf_expert_count
		self.k_clusters 		= k_clusters
		return

	def build_index_tree(self, experts):
		if len(experts) <= self.index_expert_count:
			return 	IndexedTreeNode(
						experts 	= experts,
						children 	= [],
						is_leaf  	= True
					)

		expert_indexes 	= np.concatenate([each_expert.get_index() for each_expert in experts], axis = 0)
		expert_class  	= ClusterTools.k_means_cluster(indexes = expert_indexes, clusters = self.k_clusters)
		children 		= []

		for each_cls in range(self.k_clusters):
			individual_cls_indx  	= np.where(expert_class == each_cls)[0]
			individual_cls_experts 	= list(itemgetter(*individual_cls_indx)(experts))
			child_node  			= self.build_index_tree(experts = individual_cls_experts)
			children.append(child_node)

		return 	IndexedTreeNode(
					experts 	= experts,
					children 	= children,
					is_leaf  	= False
				)

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
			self._reset_scale()
			self._index_last()

		if is_compact or is_scale:
			# Rebuild K means index tree
			self.indexed_tree 	= self.build_index_tree(experts = self.experts)

		return

	def infer(self, input_data):
		# TODO: Consult K means index tree for expert selection
		pass