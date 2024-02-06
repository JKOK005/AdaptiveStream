from Policies.Compaction.CompactionPolicy import CompactionPolicy
import tensorflow as tf
import numpy as np

class MetaCompaction(CompactionPolicy):
    def __init__(self, N, K):
        self.N = N
        self.K = K
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        
        return e_x / e_x.sum(axis=0)

    def create_model(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(self.K,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(self.K, activation='softmax')])

        return model

    def compact(self, experts, 
					  fallback_expert,
					  *args, **kwargs
				):
        new_experts = experts[-self.N:]

        K_old_experts = experts[-self.N-self.K:-self.N]

        performance = []
        for model in K_old_experts:
            result = model.loss(self.buffer.get_data(), self.buffer.get_label())
            performance.append(result * -1)
        
        weight = tf.expand_dims(self.softmax(performance), axis=-1)
        weight = tf.transpose(weight)

        # construct ground truth for predictions
        model = self.create_model()
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=loss_function)
        
        epochs = 10  # Choose the number of epochs
        x_train = tf.ones(weight.shape)
        model.fit(x_train, weight, epochs=epochs)

        pred = model(x_train)

        pred = tf.squeeze(pred)

        merged_weights = []

        for weights_list in zip(*[model_wrapper.trained_model.model.get_weights() for model_wrapper in experts[-self.N-self.K:-self.N]]):
            flag = True
            for weight, value in zip(weights_list, pred):
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
