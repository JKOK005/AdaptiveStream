from Models.ExpertEnsemble import ExpertEnsemble
from operator import itemgetter

class IndexedTreeNode(object):
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

	def get_experts(self):
		return self.experts

	def get_children(self):
		return self.children

	def is_leaf(self):
		return self.is_leaf


class IndexedExpertEnsemble(ExpertEnsemble):
	indexed_tree = None

	def __init__(self, 	index_expert_count: int, 
						*args, **kwargs
				):
		"""
		Class most shares similar features with ExpertEnsemble. 
		In addition, we build a K-means index over all fore front experts.
		This process allows search to be reduced from O(N) -> O(log N) to support high QPS for the application.

		params: index_expert_count : controls the number of experts per index tree node
		"""
		super(IndexedExpertEnsemble, self).__init__(*args, **kwargs)

		self.index_expert_count = index_expert_count
		return

	def build_index_tree(self, experts):
		if len(experts) <= self.index_expert_count:
			return 	IndexedTreeNode(
						experts 	= experts,
						children 	= [],
						is_leaf  	= True
					)

		expert_indexes 	= np.concatenate([each_expert.get_index() for each_expert in experts], 
										 axis = 0)

		for each_cls in range(self.index_expert_count):
			individual_cls_indx  	= np.where(expert_indexes == each_cls)[0]
			individual_cls_experts 	= list(itemgetter(*individual_cls_indx)(experts))



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

		if is_compact or is_scale:
			# TODO: Rebuild K means index tree
			pass

		return

	def infer(self, input_data):
		# TODO: Consult K means index tree for expert selection
		pass