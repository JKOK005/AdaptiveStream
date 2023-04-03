import logging
import numpy as np
import pandas as pd
import tensorflow as tf
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

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

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
	white_training 	 = white.sample(frac = 0.8)
	white_test  	 = white.drop(white_training.index)

	all_training 	 = pd.concat([red_training, white_training], axis = 0)
	training_feats   = tf.convert_to_tensor(all_training.drop("class", axis = 1).values, dtype = tf.float32)
	training_labels  = tf.convert_to_tensor(all_training["class"].values, dtype = tf.float32)
	training_labels  = tf.reshape(training_labels, [len(training_labels), 1])

	all_test  		 = pd.concat([red_test, white_test], axis = 0)
	test_feats  	 = tf.convert_to_tensor(all_test.drop("class", axis = 1).values, dtype = tf.float32)
	test_labels  	 = tf.convert_to_tensor(all_test["class"].values, dtype = tf.float32)
	test_labels  	 = tf.reshape(test_labels, [len(test_labels), 1])
	data_gen_test	 = tf.data.Dataset.from_tensor_slices((test_feats, test_labels))

	model  		= build_net(input_size = training_feats.shape[-1], output_size = 1)
	loss 		= losses.BinaryCrossentropy(from_logits = False)
	optimizer 	= optimizers.SGD(learning_rate = 0.001)
	model.compile(optimizer = optimizer, loss = loss)
	model.fit(training_feats, training_labels, epochs = 1000, batch_size = 64)

	preds 	= model(test_feats)
	preds  	= tf.where(preds >= 0.5, 1, 0)

	mat 	= tf.math.confusion_matrix(
	    		labels 		= tf.squeeze(test_labels),
			    predictions = tf.squeeze(preds),
			    num_classes = 2,
			)

	print(mat)