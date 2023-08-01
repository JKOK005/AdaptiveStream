from Policies.Compaction.CompactionPolicy import CompactionPolicy
import tensorflow as tf
import numpy as np
from copy import copy



class DistillCompaction(CompactionPolicy):
    def __init__(self, N, K, temperature, alpha):
        self.N = N
        self.K = K
        self.temperature = temperature
        self.alpha = alpha

    def compact(self, experts, 
					  fallback_expert,
                      buffer,
					  *args, **kwargs
				):
        new_experts = experts[-self.N:]

        K_old_experts = experts[-self.N-self.K:-self.N]

        student_model = copy(K_old_experts[0])

        student_model.trained_model.model = tf.keras.models.clone_model(K_old_experts[0].trained_model.model)
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        student_model.trained_model.model.compile(optimizer=optimizer, loss=loss_function)

        for teacher_model in K_old_experts:
            logits = teacher_model.infer(buffer.get_data())
            teacher_probs = tf.nn.softmax(logits / self.temperature, axis=-1)

            student_loss = student_model.loss(buffer.get_data(), buffer.get_label())
            student_logits = student_model.infer(buffer.get_data())
            student_probs = tf.nn.softmax(student_logits, axis=-1)
            distill_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(teacher_probs, student_logits)
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss

            student_model.trained_model.model.train_on_batch(buffer.get_data(), buffer.get_label(), sample_weight=total_loss)
        
        new_fallback_expert = tf.keras.models.clone_model(student_model.trained_model.model)
        
        return new_fallback_expert, new_experts
