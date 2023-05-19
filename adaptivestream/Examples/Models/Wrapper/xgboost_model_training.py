import logging
import tensorflow as tf
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Models.Wrapper import XGBoostModelWrapper
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor, XGBClassifier

"""
We present an example of how to train the XGBoostModelWrapper class

We will use the supervised regression task of predicting the diabetes disease progression for a patient, given a set of features.

Herein, we will follow the sample tutorial as outlined by Prasad at el. [1], 
with the only difference being the model built in Tensorflow instead of PyTorch.

[1] https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
"""

def build_net():
	params = {	
				"n_estimators" : 10000, 'min_child_weight': 10, 'learning_rate': 0.001, 'colsample_bytree': 0.3, 'max_depth': 10,
	            'subsample': 0.9, 'lambda': 0.7, 'nthread': -1, 'booster': 'gbtree', 
	            'gamma' : 0, 'eval_metric': 'mae', 'objective': 'reg:squarederror', 'seed' : 0, 'verbosity' : 1
            }
	return 	XGBRegressor(**params)

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	data 	= load_diabetes(as_frame = True)
	feats   = data["data"]
	labels  = data["target"]

	# Ensure proper formatting of all input / output tensors
	feats_as_tensor 	= tf.convert_to_tensor(feats.values, dtype = tf.float32)
	labels_as_tensor	= tf.convert_to_tensor(labels.values, dtype = tf.float32)
	labels_as_tensor 	= labels_as_tensor / max(labels_as_tensor)
	labels_as_tensor 	= tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

	# Initialize and load data into the buffer
	buffer 	= LabelledFeatureBuffer()
	buffer.add(batch_input = (feats_as_tensor, labels_as_tensor))

	base_model 	= build_net()

	model_wrapper = XGBoostModelWrapper(
						xg_boost_model 	= base_model,
						training_params = {}
					)

	model_wrapper.train(buffer = buffer)
	print(model_wrapper.infer(feats_as_tensor[0:20], predict_params = {}))