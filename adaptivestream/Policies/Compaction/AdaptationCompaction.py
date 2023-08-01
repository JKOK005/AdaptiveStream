from Policies.Compaction.CompactionPolicy import CompactionPolicy
import tensorflow as tf
import numpy as np

class AdaptationCompaction(CompactionPolicy):
    def __init__(self, N, K, strategy):
        self.N = N
        self.K = K
        self.strategy = strategy

    def compact(self, experts, 
					  fallback_expert,
                      buffer,
					  *args, **kwargs
				):
        new_experts = experts[-self.N:]

        K_old_experts = experts[-self.N-self.K:-self.N]

        if self.strategy == 'minloss':
            loss = []
            for model in K_old_experts:
                loss_value = model.loss(buffer.get_data(), buffer.get_label())
                loss.append(loss_value)
            idx = np.argmin(loss)
            new_fallback_expert = K_old_experts[idx]
        elif self.strategy == 'performance':
            evaluate = []
            for model in K_old_experts:
                performance = model.evaluate(buffer.get_data(), buffer.get_label())
                evaluate.append(perform)
            idx = np.argmax(evaluate)
            new_fallback_expert = K_old_experts[idx]
        
        return new_fallback_expert, new_experts
