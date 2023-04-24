import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from alibi_detect.models.tensorflow.losses import elbo
from matplotlib import pyplot as plt
from tensorflow.keras import layers, losses, optimizers, Sequential
from Buffer import LabelledFeatureBuffer
from Models import IndexedExpertEnsemble, IndexTreeBuilder
from Models.Router import OutlierVAERouter
from Models.Wrapper import SupervisedModelWrapper
from Rules.Scaling import BufferSizeLimit, OnlineMMDDrift
from Policies.Compaction import NoCompaction
from Policies.Scaling import NaiveScaling, NaiveKnowledgeTransfer

"""
We demonstrate an instance of scaling our mixture of expert ensemble with indexing

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
	latent_dim  = 16

	encoder_net = Sequential(
					[
						tf.keras.Input(shape = (input_size, )),
						layers.Dense(5, activation = "relu"),
						layers.Dense(2, activation = "relu"),
					]
				)

	decoder_net = Sequential(
					[
						tf.keras.Input(shape = (latent_dim, )),
						layers.Dense(2, activation = "relu"),
						layers.Dense(5, activation = "relu"),
						layers.Dense(input_size, activation = "sigmoid"),
					]
				)

	return	OutlierVAERouter(
				init_params = {
					"threshold" 	: 0.075,
					"latent_dim" 	: latent_dim,
					"encoder_net" 	: encoder_net,
					"decoder_net" 	: decoder_net,
					"samples" 		: 100,
				},

				training_params = {
					"epochs" 		: 20,
					"batch_size" 	: 64,
					"loss_fn" 		: elbo,
					"optimizer" 	: optimizers.Adam(learning_rate=5e-3),
				},

				inference_params = {
					"outlier_perc" 	: 0.80
				}
			)

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# Define scaling rules
	scaling_rules 	= 	[
							BufferSizeLimit(min_size = 256)
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

	scaling_policy  = 	NaiveKnowledgeTransfer(
							model 	= model_wrapper,
							router 	= model_router
						)

	tree_builder 	= IndexTreeBuilder(
						leaf_expert_count = 3, 
						k_clusters = 4
					)

	expert_ensemble = IndexedExpertEnsemble(
						tree_builder 		= tree_builder,
						index_dim 			= 2,
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

	for column in red.columns:
		red[column] = (red[column] - red[column].min()) / (red[column].max() - red[column].min())

	red["class"] 	 = 0
	red_training 	 = red.sample(frac = 0.8, random_state = 1)
	red_test  	 	 = red.drop(red_training.index)

	white = pd.read_csv(
	    "https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-white.csv", sep=';'
	)

	for column in white.columns:
		white[column] = (white[column] - white[column].min()) / (white[column].max() - white[column].min())

	white["class"] 	 = 1
	white_training 	 = white.sample(frac = 0.8, random_state = 1)
	white_test  	 = white.drop(white_training.index)

	all_training 	  = pd.concat([white_training, red_training], axis = 0)
	feats_as_tensor   = tf.convert_to_tensor(all_training.drop("class", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(all_training["class"].values, dtype = tf.float32)
	labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])
	data_gen_training = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	all_test  		 = pd.concat([red_test, white_test], axis = 0)
	feats_as_tensor  = tf.convert_to_tensor(all_test.drop("class", axis = 1).values, dtype = tf.float32)
	labels_as_tensor = tf.convert_to_tensor(all_test["class"].values, dtype = tf.float32)
	labels_as_tensor = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])
	data_gen_test	 = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	ingested_counts  = 0
	for batch_feats, batch_labels in data_gen_training.batch(1):
		expert_ensemble.ingest(batch_input = (batch_feats, batch_labels))
		ingested_counts += len(batch_feats)
		logging.info(f"Total data ingested: {ingested_counts}")

	# Infer on test data
	preds 	= None
	labels 	= None

	for batch_feats, batch_labels in data_gen_test.batch(1):
		pred 	= expert_ensemble.infer(input_data = batch_feats)
		preds 	= pred if preds is None else tf.concat([preds, pred], axis = 0)
		labels 	= batch_labels if labels is None else tf.concat([labels, batch_labels], axis = 0)

	preds  	= tf.where(preds >= 0.5, 1, 0)
	mat 	= tf.math.confusion_matrix(
	    		labels 		= tf.squeeze(labels),
			    predictions = tf.squeeze(preds),
			    num_classes = 2,
			)
	print(mat)

	# Plot expert indexes
	for each_expert in expert_ensemble.experts:
		expert_index = each_expert.get_index()
		expert_tags  = each_expert.get_tags()

		plt.plot(
			expert_index[0], 
			expert_index[1], 
			marker = "o", 
			markersize = 10, 
			markeredgecolor = "red"
		)

		plt.annotate(f"{expert_tags}", (expert_index[0], expert_index[1]))
	
	plt.xlabel("X") 
	plt.ylabel("Y") 		
	plt.show()