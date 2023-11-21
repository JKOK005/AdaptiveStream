import argparse
import logging
import glob
import tensorflow as tf
from adaptivestream.Buffer import LabelledFeatureBuffer
from adaptivestream.Models import ExpertEnsemble
from adaptivestream.Models.Router import OutlierVAERouter
from adaptivestream.Models.Wrapper import SupervisedModelWrapper
from adaptivestream.Models.Net import VggNet16Factory
from adaptivestream.Policies.Compaction import NoCompaction
from adaptivestream.Policies.Scaling import NaiveScaling
from adaptivestream.Policies.Checkpoint import DirectoryCheckpoint
from adaptivestream.Rules.Scaling import OnlineMMDDrift
from adaptivestream.Rules.Checkpoint import SaveOnStateChange
from Examples.Math.index_tree_creation import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Conv2DTranspose, Reshape

"""
python3 ...
"""

def build_net():
	return VggNet16Factory.get_model(input_shape = (128, 128, 3), output_size = 5)

def build_router(input_shape: (int), latent_dim: int):
	encoder_net = Sequential([
		Input(shape = input_shape),
		Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
		Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu"),
		Flatten(),
		Dense(units = latent_dim),
	])

	decoder_net = Sequential([
		Input(shape = (latent_dim)),
		Dense(units = int(input_shape[0] / 8 * input_shape[0] / 8 * 128)),
		Reshape(target_shape = (int(input_shape[0] / 8), int(input_shape[0] / 8), 128)),
		Conv2DTranspose(128, (3,3), strides = 2, padding='same', activation = "relu"),
		Conv2DTranspose(64, (3,3), strides = 2, padding='same', activation = "relu"),
		Conv2DTranspose(3, 4, strides = 2, padding='same', activation='relu')
	])

	return OutlierVAERouter(
		init_params = {
			"threshold" 	: 0.1,
		    "encoder_net" 	: encoder_net,
		    "decoder_net" 	: decoder_net,
		    "latent_dim" 	: latent_dim,
		    "samples" 		: 10
		},

		training_params = {
			"epochs" 		: 100, 
			"batch_size" 	: 128,
			"verbose" 		: True,
		}, 

		inference_params = {}
	)

def build_drift_feature_extractor(input_shape: (int), , latent_dim: int):
	encoder_net = Sequential([
		Input(shape = input_shape),
		Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
		Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu"),
		Flatten(),
		Dense(units = latent_dim),
	])
	return encoder_net

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='Linear AdaptiveStream training on Core50')
	parser.add_argument('--train_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--save_path', type = str, nargs = '?', help = 'Model checkpoint path')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# Define scaling rules
	scaling_rules 	= 	[
							OnlineMMDDrift(
								min_trigger_count = 800,
								safety_timestep = 128,
								init_params = {
									"ert" 			: 100,
									"window_size" 	: 20,
									"n_bootstraps" 	: 300,
									"preprocess_fn" : build_drift_feature_extractor(
														input_shape = (128, 128, 3), 
														latent_dim = 1024
													),
								}
							)
						]

	base_model 	 	= 	build_net()

	optimizer 		= 	tf.keras.optimizers.Adam(
							learning_rate = 0.001,
						)

	loss_fn 		=  	tf.keras.losses.SparseCategoricalCrossentropy(
							reduction = tf.keras.losses.Reduction.SUM
						)
						
	model_wrapper 	= 	SupervisedModelWrapper(
							base_model 		= base_model,
							optimizer 		= optimizer,
							loss_fn 		= loss_fn,
							training_params = {}, 
						)

	model_router  	= 	build_router(input_shape = (128, 128, 3), latent_dim = 1024)

	scaling_policy  = 	NaiveScaling(
							model 	= model_wrapper,
							router 	= model_router
						)

	# Define checkpoint rules and policies
	checkpoint_rules = 	[ SaveOnStateChange() ]

	checkpoint_policy = DirectoryCheckpoint(save_path = args.save_path)

	expert_ensemble = ExpertEnsemble(
						buffer 				= LabelledFeatureBuffer(),
						scaling_rules 		= scaling_rules,
						scaling_policy 		= scaling_policy, 
						compaction_rules 	= [],
						compaction_policy 	= NoCompaction(),
						checkpoint_rules 	= checkpoint_rules,
						checkpoint_policy 	= checkpoint_policy
					)

	# for file in glob.glob(f"{args.train_dir}/*.csv"):
	# 	train_df = pd.read_csv(file)
	# 	train_df = train_df.sample(frac = 1)

	# 	feats_as_tensor   = tf.convert_to_tensor(train_df.drop("price", axis = 1).values, dtype = tf.float32)
	# 	labels_as_tensor  = tf.convert_to_tensor(train_df["price"].values, dtype = tf.float32)
	# 	labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])
	# 	data_gen_training = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	# 	ingested_counts  = 0
	# 	for batch_feats, batch_labels in data_gen_training.batch(64):
	# 		batch_feats_wo_ts 	= batch_feats
	# 		expert_ensemble.ingest(batch_input = (batch_feats_wo_ts, batch_labels))
			
	# 		ingested_counts += len(batch_feats)
	# 		logging.info(f"Total data ingested: {ingested_counts}, cur file: {file}")

	# if expert_ensemble.buffer.get_count() > 0:
	# 	expert_ensemble.scale_experts()
	# 	expert_ensemble.checkpoint_policy.save(expert_emsemble = expert_ensemble, log_state = True)