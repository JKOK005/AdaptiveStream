import logging
import tensorflow as tf
from alibi_detect.models.tensorflow.losses import elbo
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Models.Router.OutlierVAERouter import OutlierVAERouter
from Models.Wrapper.SupervisedModelWrapper import SupervisedModelWrapper
from Policies.Scaling.NaiveScaling import NaiveScaling
from sklearn.datasets import load_diabetes
from tensorflow.keras import layers, losses, optimizers, Sequential

"""
We show how to incorporate the OutlierVAERouter router into each expert when training on the diabetes dataset from sklearn
The encoder-decoder network composition of the router can be found in [1]

We use a simple feed forward neural net as base model for each expert

[1]: https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/vae.html
"""

def build_router(input_size):
	encoder_net = Sequential(
					[
						tf.keras.Input(shape = (input_size, )),						
						layers.Dense(5, activation = "relu"),
						layers.Dense(2, activation = "relu"),
					]
				)

	decoder_net = Sequential(
					[
						tf.keras.Input(shape = (16, )),
						layers.Dense(2, activation = "relu"),
						layers.Dense(5, activation = "relu"),
						layers.Dense(input_size),
					]
				)

	return	OutlierVAERouter(
				init_params = {
					"threshold" 	: 0.1,
					"encoder_net" 	: encoder_net,
					"decoder_net" 	: decoder_net,
					"latent_dim" 	: 16,
					"samples" 		: 10 
				},

				training_params = {
					"loss_fn" 		: elbo,
					"optimizer" 	: optimizers.Adam(learning_rate=1e-3),
				},

				inference_params = {
					"outlier_perc" 	: 80
				}
			)

def build_model_wrapper(input_size, output_size):
	base_model  = Sequential(
					[
						tf.keras.Input(shape = (input_size, )),
						layers.Dense(5, activation = "relu"),
						layers.Dense(output_size, activation = "relu")
					]
				)
	criterion	= losses.MeanSquaredError()
	optimizer 	= optimizers.SGD(learning_rate = 0.0001)
	return 	SupervisedModelWrapper(
				base_model 		= base_model,
				optimizer 		= optimizer,
				loss 			= criterion,
				training_params = {
					"epochs" 		: 1500,
					"batch_size" 	: 64
				}
			)

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	data 	= load_diabetes(as_frame = True)
	feats   = data["data"]
	labels  = data["target"]

	# Ensure proper formatting of all input / output tensors
	feats_as_tensor 	= tf.convert_to_tensor(feats.values, dtype = tf.float32)[:-2]
	labels_as_tensor	= tf.convert_to_tensor(labels.values, dtype = tf.float32)[:-2]
	labels_as_tensor 	= tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

	# Initialize and load data into the buffer
	buffer 	= LabelledFeatureBuffer()
	buffer.add(batch_input = (feats_as_tensor, labels_as_tensor))

	model_wrapper 	= build_model_wrapper(input_size = feats_as_tensor.shape[1], output_size = 1)
	router 			= build_router(input_size = feats_as_tensor.shape[1])

	# Train a single expert using naive scaling
	policy  		= NaiveScaling(model = model_wrapper, router = router)
	policy.set_buffer(buffer = buffer)
	new_expert 		= policy.train_expert()

	# Perform outlier detection
	test_data  		= feats_as_tensor[-1, None]
	rand_data 		= tf.random.uniform([1, feats_as_tensor.shape[1]])
	
	print(f"Test data passes router selection: {new_expert.permit_entry(input_X = test_data)}")
	print(f"Random data passes router selection: {new_expert.permit_entry(input_X = rand_data)}")