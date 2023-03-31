import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from Buffer import LabelledFeatureBuffer
from Models import ExpertEnsemble
from Models.Router import OneClassSVMRouter
from Models.Wrapper import SupervisedModelWrapper
from Rules.Scaling import BufferSizeLimit, OnlineMMDDrift
from Policies.Compaction import NoCompaction
from Policies.Scaling import NaiveScaling
from tensorflow.keras import layers, losses, optimizers, Sequential

"""
We demonstrate an instance of scaling our mixture of expert ensemble 

In this example, we will load data into our buffer in batches
We will define a scaling rule which requests the training and provision of a new expert after exceeding a buffer threshold size
Thereafter, we will continuously load data into our buffer via the ExeperEnsemble ingest method 

We simulate these events using the wine quality dataset [1]
Our task is to predict, from a given feature set, if a sample data shown is white or red wine

[1] : https://docs.seldon.io/projects/alibi-detect/en/stable/examples/cd_online_wine.html
"""

def build_net(input_size, output_size):
	return 	Sequential(
				[
					tf.keras.Input(shape = (input_size, )),
					layers.Dense(5, activation = "relu"),
					layers.Dense(output_size, activation = "sigmoid")
				]
			)

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

	# Define scaling rules
	scaling_rules 	= 	[
							OnlineMMDDrift(
								min_trigger_count 	= 1024,
								safety_timestep 	= 10,
								init_params 		= {
									"ert" 			: 150,
									"window_size" 	: 5,
									"n_bootstraps" 	: 2000,
								}
							),

							BufferSizeLimit(min_size = 4096)
						]

	# Define scaling policy
	model_wrapper 	= 	SupervisedModelWrapper(
							base_model 		= build_net(12, 1),
							optimizer 		= optimizers.SGD(learning_rate = 0.001),
							loss 			= losses.BinaryCrossentropy(from_logits = False),
							training_params = {
								"epochs" 		: 1000,
								"batch_size" 	: 64
							},
						)

	model_router  	= 	build_router(input_size = 12)

	scaling_policy  = 	NaiveScaling(
							model 	= model_wrapper,
							router 	= model_router
						)

	expert_ensemble = ExpertEnsemble(
						buffer 				= LabelledFeatureBuffer(),
						scaling_rules 		= scaling_rules,
						scaling_policy 		= scaling_policy, 
						compaction_rules 	= [],
						compaction_policy 	= NoCompaction(),
					)

	# Mix red & white wine datasets together, with white = 1 & red = 0
	red = pd.read_csv(
    	"https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-red.csv", sep=';'
	)
	red = red.sample(frac = 1)
	red["class"] = 0

	white = pd.read_csv(
	    "https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-white.csv", sep=';'
	)
	white = white.sample(frac = 1)
	white["class"] 	= 1

	all_wines = pd.concat([red, white], axis = 0)

	feats_as_tensor  = tf.convert_to_tensor(all_wines.drop("class", axis = 1).values, dtype = tf.float32)
	labels_as_tensor = tf.convert_to_tensor(all_wines["class"].values, dtype = tf.float32)
	labels_as_tensor = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])
	data_gen 		 = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	ingested_counts  = 0
	for batch_feats, batch_labels in data_gen.batch(1):
		expert_ensemble.ingest(batch_input = (batch_feats, batch_labels))
		ingested_counts += len(batch_feats)
		logging.info(f"Total data ingested: {ingested_counts}")

	# Infer on test data
	# TODO: Implement inference