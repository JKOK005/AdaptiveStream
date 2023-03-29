import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Rules.Scaling import OnlineMMDDrift
from sklearn.datasets import load_diabetes

"""
We will use the wine quality dataset to test the effectiveness of our drift detector
This experiment is based on the experiment in [1]

Interestingly, we are getting really strong perforrmance on the drift detector for the dataset when we select a good safety_timestep value.
For this example, when safety_timestep = 20, non-drift datasets were not flagged; only drift datasets were flagged.

[1] : https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_online_wine.html
"""

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	red = pd.read_csv(
    	"https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-red.csv", sep=';'
	)
	red = np.asarray(red, np.float32)

	white = pd.read_csv(
	    "https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-white.csv", sep=';'
	)
	white = np.asarray(white, np.float32)

	# Ensure proper formatting of all input / output tensors
	# Since this is an unsupervised detection task, we ignore the values of the labels
	white_feats_as_tensor 	= tf.convert_to_tensor(white, dtype = tf.float32)
	white_labels_as_tensor  = tf.ones([white_feats_as_tensor.shape[0], 1])

	red_feats_as_tensor  	= tf.convert_to_tensor(red, dtype = tf.float32)
	red_labels_as_tensor  	= tf.ones([red_feats_as_tensor.shape[0], 1])

	training_feats  	= white_feats_as_tensor[:-500]
	training_labels  	= white_labels_as_tensor[:-500]

	# Initialize and load data into the buffer
	buffer 	= LabelledFeatureBuffer()
	buffer.add(batch_input = (training_feats, training_labels))

	mmd_drift = OnlineMMDDrift(	min_trigger_count = 3000,
								safety_timestep = 20,
								init_params = {
									"ert" 			: 100,
									"window_size" 	: 50,
									"n_bootstraps" 	: 2000,
								}
							)

	# First call to train the detector
	mmd_drift.check_scaling(buffer = buffer)

	for seed in range(20):
		non_drift_feats  	= tf.random.shuffle(white_feats_as_tensor[-500:], seed = seed)
		non_drift_labels  	= white_labels_as_tensor[-500:]

		drift_feats  		= tf.random.shuffle(red_feats_as_tensor[0:500], seed = seed)
		drift_labels  		= red_labels_as_tensor[0:500]

		drift_time = 0
		mmd_drift.drift_model.reset_state()

		# Subsequent calls to measure drift
		for i in range(len(non_drift_feats)):
			buffer.add(batch_input = (non_drift_feats[i : i+1], non_drift_labels[i : i+1]))
			is_drift = mmd_drift.check_scaling(buffer = buffer)
			drift_time += 1
			if is_drift:
				break
		print(f"Non-drift data detected in {drift_time} timesteps")

		# Now check for dataset with drift
		drift_time = 0
		mmd_drift.drift_model.reset_state()
		
		# Subsequent calls to measure drift
		for i in range(len(drift_feats)):
			buffer.add(batch_input = (drift_feats[i : i+1], drift_labels[i : i+1]))
			is_drift = mmd_drift.check_scaling(buffer = buffer)
			drift_time += 1
			if is_drift:
				break
		print(f"Drift data detected in {drift_time} timesteps")