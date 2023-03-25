import logging
import tensorflow as tf
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Wrapper.SupervisedModelWrapper import SupervisedModelWrapper
from sklearn.datasets import load_diabetes
from tensorflow.keras import layers, losses, optimizers, Sequential

"""
We present an example of how to train the SupervisedModelWrapper class

We will use the supervised regression task of predicting the diabetes disease progression for a patient, given a set of features.

Herein, we will follow the sample tutorial as outlined by Prasad at el. [1], 
with the only difference being the model built in Tensorflow instead of PyTorch.

[1] https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
"""

def build_net(input_size, output_size):
	return 	Sequential(
				[
					tf.keras.Input(shape = (input_size, )),
					layers.Dense(5, activation = "relu"),
					layers.Dense(output_size, activation = "relu")
				]
			)

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	data 	= load_diabetes(as_frame = True)
	feats   = data["data"]
	labels  = data["target"]

	# Ensure proper formatting of all input / output tensors
	feats_as_tensor 	= tf.convert_to_tensor(feats.values, dtype = tf.float32)
	labels_as_tensor	= tf.convert_to_tensor(labels.values, dtype = tf.float32)
	labels_as_tensor 	= tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

	# Initialize and load data into the buffer
	buffer 	= LabelledFeatureBuffer()
	buffer.add(batch_input = (feats_as_tensor, labels_as_tensor))

	base_model 	= build_net(feats_as_tensor.shape[1], 1)
	criterion	= losses.MeanSquaredError()
	optimizer 	= optimizers.SGD(learning_rate = 0.001)

	model_wrapper = SupervisedModelWrapper(
						base_model 	= base_model,
						optimizer 	= optimizer,
						loss 		= criterion
					)

	model_wrapper.train(
		buffer 		= buffer,
		epoch  		= 1500,
		batch_size 	= 64, 
	)