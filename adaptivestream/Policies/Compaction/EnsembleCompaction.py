from Policies.Compaction.CompactionPolicy import CompactionPolicy
import tensorflow as tf

class EnsembleCompaction(CompactionPolicy):
    def __init__(self, N, K, strategy):
        self.N = N
        self.K = K
        self.strategy = strategy
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        
        return e_x / e_x.sum(axis=0)

    def compact(self, experts, 
					  fallback_expert,
                      buffer,
					  *args, **kwargs
				):
        
        new_experts = experts[-self.N:]
        
        if self.strategy == 'latest':
            new_fallback_expert = experts[-self.N-1]
        elif self.strategy == 'oldest':
            new_fallback_expert = experts[-self.N-self.K]
        elif self.strategy == 'merge':
            merged_weights = []

            for weights_list in zip(*[model_wrapper.trained_model.model.get_weights() for model_wrapper in experts[-self.N-self.K:-self.N]]):
                merged_weights.append(tf.reduce_sum(weights_list, axis=0))
            
            new_fallback_expert = tf.keras.models.clone_model(experts[0].trained_model.model)
            new_fallback_expert.set_weights(merged_weights)
        elif self.strategy == 'weighted-merge':
            performance = []

            for model in experts[-self.N-self.K:-self.N]:
                result = model.evaluate(buffer.get_data(), buffer.get_label())
                performance.append(result)
            
            weight = self.softmax(performance)

            merged_weights = []

            for weights_list in zip(*[model_wrapper.trained_model.model.get_weights() for model_wrapper in experts[-self.N-self.K:-self.N]]):
                flag = True
                for weight, value in zip(weights_list, weight):
                if flag:
                    temp_weight = weight * value
                    flag = False
                else:
                    temp_weight += weight * value
            temp_weight = temp_weight / self.K
            merged_weights.append(temp_weight)
            
            new_fallback_expert = tf.keras.models.clone_model(experts[0].trained_model.model)
            new_fallback_expert.set_weights(merged_weights)
        
        return new_fallback_expert, new_experts
