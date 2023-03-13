class ModelManager(object):
	self.experts 				= [] 	# Each expert is a {"model" : expert, "gate" : router} dict
	self.fallback_expert 		= None

	self.scaling_policy 		= None 
	self.compaction_strategy 	= None
	self.buffer 				= None

	def __init__(self, 	scaling_policy, 
						compaction_strategy,
						buffer
				):
		"""
		Policy returns a single trained (router, model) pair
		"""
		self.scaling_policy = scaling_policy
		self.compaction_strategy = compaction_strategy
		self.buffer = buffer
		return

	def train(self, batch_input_data):
		(router, expert) = self.scaling_policy.train_on_input(batch_input_data)
		return (router, expert)

	def infer(self, input_data):
		for each_expert in self.experts.reverse():
			if each_expert.select_as_route(input_data):
				return each_expert.infer(input_data)
		return self.fallback_expert.infer(input_data)

	def ingest(self, batch_input_data):
		self.buffer.add(input_data)
		
		if self.buffer.start_training():
			(router, expert) = self.train(batch_input_data = self.buffer.get_buffer())

			if self.compaction_strategy.start_compacting(self.experts):
				(new_fallback_expert, new_experts) = self.compaction_strategy.compact(self.experts, fallback_expert)
				self.fallback_expert = new_fallback_expert
				self.experts = new_experts
		
		self.buffer.clear()
		return
