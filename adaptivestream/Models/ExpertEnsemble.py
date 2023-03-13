class ExpertEnsemble(object):
	self.experts 			= []
	self.fallback_expert 	= None

	self.scaling_policy 	= None 
	self.compaction_policy 	= None
	self.buffer 			= None

	def __init__(self, 	scaling_policy, 
						compaction_policy,
						buffer
				):
		"""
		Policy returns a single trained (router, model) pair
		"""
		self.scaling_policy = scaling_policy
		self.compaction_policy = compaction_policy
		self.buffer = buffer
		return

	def _train(self, batch_input):
		(router, expert) = self.scaling_policy.train_on_input(batch_input)
		return (router, expert)

	def infer(self, input_data):
		for each_expert in self.experts.reverse():
			if each_expert.permit_entry(input_data):
				return each_expert.infer(input_data)
		return self.fallback_expert.infer(input_data)

	def ingest(self, batch_input):
		self.buffer.add(batch_input)
		
		if self.buffer.exceed_capacity():
			(router, expert) = self.train(batch_input = self.buffer.get_buffer())

			if self.compaction_policy.start_compacting(self.experts):
				(new_fallback_expert, new_experts) = self.compaction_policy.compact(self.experts, fallback_expert)
				self.fallback_expert = new_fallback_expert
				self.experts = new_experts
		
		self.buffer.clear()
		return
