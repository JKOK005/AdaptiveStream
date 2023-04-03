from Policies.Compaction.CompactionPolicy import CompactionPolicy
import tensorflow as tf

class EnsembleCompaction(CompactionPolicy):
    def __init__(self, N, K, strategy):
        self.N = N
        self.K = K
        self.strategy = strategy

    def compact(self, experts, 
					  fallback_expert,
					  *args, **kwargs
				):
        assert (self.N + self.K < len(experts)), 'The number of experts is not full.'
        
        new_experts = experts[-self.N:]
        
        if self.strategy == 'latest':
            new_fallback_expert = experts[-self.N-1]
        elif self.strategy == 'oldest':
            new_fallback_expert = experts[-self.N-self.K]
        elif self.strategy == 'merge':
            merged_weights = []

            for weights_list in zip(*[model.get_weights() for model in experts[-self.N-self.K:-self.N]]):
                merged_weights.append(tf.reduce_sum(weights_list, axis=0))
            
            new_fallback_expert = tf.keras.models.clone_model(experts[0])
            new_fallback_expert.set_weights(merged_weights)

		return new_fallback_expert, new_experts
