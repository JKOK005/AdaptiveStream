import argparse
import logging
import glob
import tensorflow as tf
from adaptivestream.Buffer import LabelledFeatureBuffer
from adaptivestream.Models import ExpertEnsemble
from adaptivestream.Models.Router import OutlierAERouter
from adaptivestream.Models.Wrapper import SupervisedModelWrapper
from adaptivestream.Models.Net import VggNet16Factory
from adaptivestream.Policies.Compaction import NoCompaction
from adaptivestream.Policies.Scaling import NaiveScaling
from adaptivestream.Policies.Checkpoint import DirectoryCheckpoint
from adaptivestream.Policies.Compaction import LastRemovalCompaction
from adaptivestream.Rules.Scaling import OnlineMMDDrift, BufferSizeLimit
from adaptivestream.Rules.Checkpoint import SaveOnStateChange
from adaptivestream.Rules.Compaction import SizeRules
from Examples.Math.index_tree_creation import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Conv2DTranspose, Reshape
from tqdm import tqdm

"""
python3 pipeline/Core50/Linear_adaptive.py \
--train_dir /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/train \
--test_dir /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/test \
--save_path checkpoint/core50/vgg/linear
"""

def build_net():
	return VggNet16Factory.get_model(input_shape = (128, 128, 3,), output_size = 10)

def build_router(input_shape: (int), latent_dim: int):
	net = Sequential([
		Input(shape = input_shape),
		Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation = "relu"),
		Conv2D(filters = 8, kernel_size = (3,3), padding = "same", activation = "relu"),
		Flatten(),
		Dense(units = latent_dim),
		Dense(units = int(input_shape[0] / 8 * input_shape[0] / 8 * 32)),
		Reshape(target_shape = (int(input_shape[0] / 8), int(input_shape[0] / 8), 32)),
		Conv2DTranspose(32, (3,3), strides = 2, padding='same', activation = "relu"),
		Conv2DTranspose(8, (3,3), strides = 2, padding='same', activation = "relu"),
		Conv2DTranspose(3, 4, strides = 2, padding='same', activation='relu')
	])

	return OutlierAERouter(
		init_params = {
			"threshold" 	: 0.1,
		    "ae" 			: net,
		    "data_type" 	: "image",
		},

		training_params = {
			"epochs" 		: 2, 
			"batch_size" 	: 32,
			"verbose" 		: False,
		}, 

		inference_params = {
		}
	)

def build_drift_feature_extractor(input_shape: (int), latent_dim: int):
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
	parser.add_argument('--test_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--save_path', type = str, nargs = '?', help = 'Model checkpoint path')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# Define scaling rules
	# scaling_rules 	= 	[
	# 						OnlineMMDDrift(
	# 							min_trigger_count = 32,
	# 							safety_timestep = 256,
	# 							init_params = {
	# 								"ert" 			: 256,
	# 								"window_size" 	: 20,
	# 								"n_bootstraps" 	: 300,
	# 								"preprocess_fn" : build_drift_feature_extractor(
	# 													input_shape = (128, 128, 3,), 
	# 													latent_dim = 1024
	# 												),
	# 							}
	# 						)
	# 					]

	scaling_rules 	= [ BufferSizeLimit(min_size = 64) ]

	base_model 	 	= 	build_net()

	optimizer 		= 	tf.keras.optimizers.legacy.Adam(
							learning_rate = 0.001,
						)

	loss_fn 		=  	tf.keras.losses.SparseCategoricalCrossentropy(
							reduction = tf.keras.losses.Reduction.SUM
						)
						
	model_wrapper 	= 	SupervisedModelWrapper(
							base_model 		= base_model,
							optimizer 		= optimizer,
							loss_fn 		= loss_fn,
							training_params = {
								"batch_size" : 64,
								"epochs" : 2,
							}, 
						)

	model_router  	= 	build_router(input_shape = (128, 128, 3,), latent_dim = 64)

	scaling_policy  = 	NaiveScaling(
							model 	= model_wrapper,
							router 	= model_router
						)

	# Define compaction rules
	compaction_rules 	= [ SizeRules(0, 45) ]

	compaction_policy 	= LastRemovalCompaction()

	# Define checkpoint rules and policies
	# checkpoint_rules = 	[ SaveOnStateChange() ]
	checkpoint_rules = 	[]

	checkpoint_policy = DirectoryCheckpoint(save_path = args.save_path)

	expert_ensemble = ExpertEnsemble(
						buffer 				= LabelledFeatureBuffer(),
						scaling_rules 		= scaling_rules,
						scaling_policy 		= scaling_policy, 
						compaction_rules 	= compaction_rules,
						compaction_policy 	= compaction_policy,
						checkpoint_rules 	= checkpoint_rules,
						checkpoint_policy 	= checkpoint_policy
					)

	for each_file in glob.glob(f"{args.train_dir}/*.npy"):
		train_dat 	= np.load(each_file, allow_pickle = True) # load
		np.random.shuffle(train_dat)

		for each_training_dat in tqdm(np.array_split(train_dat, len(train_dat) // 64)):
			feats_as_tensor   = tf.convert_to_tensor(each_training_dat[:,0].tolist(), dtype = tf.float32)
			labels_as_tensor  = tf.convert_to_tensor(each_training_dat[:,1].tolist(), dtype = tf.float32)
			labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

			ingested_counts  = 0
			expert_ensemble.ingest(batch_input = (feats_as_tensor, labels_as_tensor))
			ingested_counts += len(feats_as_tensor)

		logging.info(f"Total data ingested: {ingested_counts}, cur file: {each_file}")

	if expert_ensemble.buffer.get_count() > 0:
		expert_ensemble.scale_experts()
		expert_ensemble.checkpoint_policy.save(expert_emsemble = expert_ensemble, log_state = True)
