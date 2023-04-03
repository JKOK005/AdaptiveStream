from Policies.Compaction.CompactionPolicy import CompactionPolicy
import tensorflow as tf
import numpy as np

class EnsembleCompaction(CompactionPolicy):
    def __init__(self, N, K, strategy):
        self.N = N
        self.K = K
        self.strategy = strategy

    def compact(self, experts, 
					  fallback_expert,
					  *args, **kwargs
				):
        assert (self.N + self.K != len(experts)), 'The number of experts is not full.'
        new_experts = experts[-self.N:]

        K_old_experts = experts[-self.N-self.K:-self.N]

        if strategy == 'minloss':
            loss = []
            for model in K_old_experts:
                loss_value = model.loss(self.buffer.get_data, self.buffer.get_label)
                loss.append(loss_value)
            idx = np.argmin(loss)
            new_fallback_expert = K_old_experts[idx]
        else:
            # TODO
            new_fallback_expert = K_old_experts[0]
		return new_fallback_expert, new_experts
