class ExpertEnsemble(object):
	experts 			= []
	fallback_expert 	= None

	scaling_policy 		= None 
	compaction_policy 	= None
	buffer 				= None

	def __init__(self, 	scaling_rules,
						scaling_policy, 
						compaction_rules,
						compaction_policy,
						buffer,
						*args, **kwargs
				):
		"""
		Policy returns a single trained (router, model) pair
		"""
		self.scaling_rules 		= scaling_rules
		self.scaling_policy 	= scaling_policy
		self.compaction_rules 	= compaction_rules
		self.compaction_policy 	= compaction_policy
		self.buffer 			= buffer

		self.scaling_policy.set_buffer(buffer = buffer)
		self.compaction_policy.set_buffer(buffer = buffer)
		return

	def _check_to_scale(self):
		def check(rule):
			if type(rule) == tuple:
				return all([check(rule = each_rule) for each_rule in rule])
			else:
				return rule.check_scaling(buffer = self.buffer)

		return any([check(rule = each_rule) for each_rule in self.scaling_rules])

	def _check_to_compact(self):
		def check(rule):
			if type(rule) == tuple:
				return all([check(rule = each_rule) for each_rule in rule])
			else:
				return rule.check_compact(experts = self.experts)

		decisions 	= [check(rule = each_rule) for each_rule in self.compaction_rules]
		return any(decisions)

	def ingest(self, batch_input):
		self.buffer.add(batch_input = batch_input)
			
		if self._check_to_compact():
			(new_fallback_expert, new_experts) 	= self.compaction_policy.compact(
													expert_chain = self.experts, 
													prev_fallback_expert = self.fallback_expert, 
												)

			self.fallback_expert = new_fallback_expert
			self.experts = new_experts

		if self._check_to_scale():
			expert 	= self.self.scaling_policy.train_expert()
			self.experts.append(expert)
			self.buffer.clear()
		return

	def infer(self, input_data):
		for each_expert in self.experts.reverse():
			if each_expert.bypass(input_data):
				return each_expert.infer(input_data)
		return self.fallback_expert.infer(input_data)