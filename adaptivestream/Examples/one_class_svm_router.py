import logging
import tensorflow as tf
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Models.Router.OneClassSVMRouter import OneClassSVMRouter
from sklearn.datasets import load_diabetes

"""
We present an example of how to train the OneClassSVMRouter class

We will use the features from the diabetes dataset found in sklearn.datasets
"""

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

	router 	= OneClassSVMRouter(nu = 0.1, kernel = "rbf", gamma = 0.1)
	router.train(buffer = buffer)

	router.permit_entry(input_X = feats_as_tensor)