import logging
import tensorflow as tf
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Models.Router.OneClassSVMRouter import OneClassSVMRouter
from sklearn.datasets import load_diabetes

"""
We show how to train the OneClassSVMRouter router on the diabetes dataset from sklearn

Parameters for the OneClassSVMRouter can be found at [1]

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
"""

def build_router(input_size):
	return	OneClassSVMRouter(
				init_params = {
					"kernel" 	: "rbf",
					"degree" 	: 3,
					"max_iter" 	: 1000
				},
			)

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	data 	= load_diabetes(as_frame = True)
	feats   = data["data"]
	labels  = data["target"]

	# Ensure proper formatting of all input / output tensors
	feats_as_tensor 	= tf.convert_to_tensor(feats.values, dtype = tf.float32)[:-1]
	labels_as_tensor	= tf.convert_to_tensor(labels.values, dtype = tf.float32)[:-1]
	labels_as_tensor 	= tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

	# Initialize and load data into the buffer
	buffer 	= LabelledFeatureBuffer()
	buffer.add(batch_input = (feats_as_tensor, labels_as_tensor))

	router = build_router(input_size = feats_as_tensor.shape[1])
	router.train(buffer = buffer)

	# Perform outlier detection
	test_data  		= feats_as_tensor[-1, None]
	rand_data 		= tf.random.uniform([1, feats_as_tensor.shape[1]])
	
	print(f"Test data passes router selection: {router.permit_entry(input_X = test_data)}")
	print(f"Random data passes router selection: {router.permit_entry(input_X = rand_data)}")