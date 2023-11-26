import logging
import tensorflow as tf
import uuid

class ExpertEnsemble(object):
	compaction_rules 	= None
	checkpoint_rules 	= None
	
	scaling_policy 		= None 
	compaction_policy 	= None
	checkpoint_policy 	= None
	
	buffer 				= None
	state  				= {"id" : uuid.uuid4().hex, "range" : None}

	def __init__(self, 	scaling_rules,
						scaling_policy, 
						compaction_rules,
						compaction_policy,
						buffer,
						checkpoint_rules 	= [],
						checkpoint_policy 	= None,
						*args, **kwargs
				):
		"""
		Policy returns a single trained (router, model) pair
		"""
		self.experts 			= []
		self.fallback_expert 	= None
		self.scaling_rules 		= scaling_rules
		self.scaling_policy 	= scaling_policy
		self.compaction_rules 	= compaction_rules
		self.compaction_policy 	= compaction_policy
		self.checkpoint_rules 	= checkpoint_rules
		self.checkpoint_policy  = checkpoint_policy
		self.buffer 			= buffer
		self.logger  			= logging.getLogger("ExpertEnsemble")

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

	def _reset_scale(self):
		for each_rule in self.scaling_rules:
			each_rule.reset()

		self.state = {"id" : uuid.uuid4().hex, "range" : self.buffer.get_batch_timestamps()}
		self.scaling_policy.reset()
		self.buffer.clear()
		return

	def _check_to_compact(self):
		def check(rule):
			if type(rule) == tuple:
				return all([check(rule = each_rule) for each_rule in rule])
			else:
				return rule.check_compaction(experts = self.experts)
		decisions 	= [check(rule = each_rule) for each_rule in self.compaction_rules]
		return any(decisions)

	def _check_to_checkpoint(self):
		def check(rule):
			if type(rule) == tuple:
				return all([check(rule = each_rule) for each_rule in rule])
			else:
				return rule.check_checkpoint(expert_ensemble = self)
		decisions 	= [check(rule = each_rule) for each_rule in self.checkpoint_rules]
		return any(decisions)

	def get_state(self):
		return self.state

	def scale_experts(self):
		expert = self.scaling_policy.train_expert()
		if self.fallback_expert is not None:
			self.experts.append(expert)
		else:
			self.fallback_expert = expert
		return

	def ingest(self, batch_input,
					 batch_timestamp: int = -1,
			):
		
		self.buffer.add(batch_input = batch_input, 	
						batch_timestamp = batch_timestamp)

		print(f"Buffer size: {self.buffer.get_count()}")

		is_compact 	= self._check_to_compact()
		is_scale  	= self._check_to_scale()
			
		if is_compact:
			(new_fallback_expert, new_experts) 	= self.compaction_policy.compact(
													experts = self.experts, 
													fallback_expert = self.fallback_expert, 
												)

			self.fallback_expert = new_fallback_expert
			self.experts = new_experts

		if is_scale:
			self.scale_experts()
			self._reset_scale()

		if self._check_to_checkpoint():
			self.checkpoint_policy.save(expert_emsemble = self, log_state = True)
		return

	def infer(self, input_data):
		probs  		= [each_expert.prob(input_data) for each_expert in self.experts]
		max_probs 	= max(probs)
		
		if max_probs >= 0.25:
			best_indx   = probs.index(max_probs)
			best_expert = self.experts[best_indx]
		else:
			# No experts achieve at least 25% confidence in infering current batch
			best_expert = self.fallback_expert
		return best_expert.infer(input_data)

	def infer_w_smpls(self, input_data,
							truth_smpls,
							alpha = 0.1,
					):
		"""	
		Ground truth samples are used to compliment outlier scores for expert selection. 
		
		We fist obtain the loss over ground truth samples for all experts (loss_ground). 
		Thereafter, we define a score S = alpha * softmax(outlier_score) + (1 - alpha) * softmax(loss_ground).

		By virtue of the fact that a lower outlier_score / loss_ground implies better fit of the data, 
		a lower value of S implies a more suitable expert for selection.
		"""
		all_experts 				= [self.fallback_expert] + self.experts
		(truth_feats, truth_labels) = truth_smpls

		probs  		= [each_expert.prob(input_data) for each_expert in self.experts]
		probs 		= [0.5 - min(max(probs), 0.5)] + probs
		probs 		= tf.cast(probs, tf.float32)

		loss 		= [each_expert.loss(input_X = truth_feats, input_y = truth_labels) for each_expert in all_experts]
		loss_sm  	= tf.math.softmax(tf.math.log(loss))
		loss_sm 	= tf.cast(loss_sm, tf.float32)

		agg_score  	= alpha * (1 - probs) + (1 - alpha) * loss_sm
		best_indx   = int(tf.argmin(agg_score))
		print(f"Best expert index: {best_indx}, experts: {len(all_experts)}")
		return all_experts[best_indx].infer(input_data)