import logging
import tensorflow as tf
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Rules.Scaling import OnlineMMDDrift
from sklearn.datasets import load_diabetes

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	data 	= load_diabetes(as_frame = True)
	feats   = data["data"]
	labels  = data["target"]

	# Ensure proper formatting of all input / output tensors
	feats_as_tensor 	= tf.convert_to_tensor(feats.values, dtype = tf.float32)
	labels_as_tensor	= tf.convert_to_tensor(labels.values, dtype = tf.float32)
	labels_as_tensor 	= tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

	training_feats  	= feats_as_tensor[:350]
	training_labels  	= labels_as_tensor[:350]

	non_drift_feats  	= feats_as_tensor[350:]
	non_drift_labels  	= labels_as_tensor[350:]

	# Initialize and load data into the buffer
	buffer 	= LabelledFeatureBuffer()
	buffer.add(batch_input = (training_feats, training_labels))

	mmd_drift = OnlineMMDDrift(	min_trigger_count = 300,
								safety_timestep = 200,
								init_params = {
									"ert" 			: 30,
									"window_size" 	: 30,
									"n_bootstraps" 	: 1000,
								}
							)

	# First call to train the detector
	mmd_drift.check_scaling(buffer = buffer)

	# Subsequent calls to measure drift
	for i in range(len(non_drift_feats)):
		buffer.add(batch_input = (training_feats[i : i+1], training_labels[i : i+1]))
		is_drift = mmd_drift.check_scaling(buffer = buffer)
		print(is_drift)