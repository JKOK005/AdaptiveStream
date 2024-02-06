import argparse
import logging
import glob
import pandas as pd
import tensorflow as tf
from adaptivestream.Buffer import LabelledFeatureBuffer
from adaptivestream.Models import ExpertEnsemble, IndexedExpertEnsemble, IndexTreeBuilder
from adaptivestream.Models.Router import IsolationForestRouter
from adaptivestream.Models.Wrapper import SupervisedModelWrapper
from adaptivestream.Policies.Compaction import NoCompaction
from adaptivestream.Policies.Scaling import NaiveScaling
from adaptivestream.Policies.Checkpoint import DirectoryCheckpoint
from adaptivestream.Policies.Compaction import EnsembleCompaction
from adaptivestream.Rules.Scaling import OnlineMMDDrift, BufferSizeLimit
from adaptivestream.Rules.Checkpoint import SaveOnStateChange
from adaptivestream.Rules.Compaction import SizeRules
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

"""
python3 pipeline/Airbnb_compaction/airbnb_static.py \
--train_dir data/airbnb/apac/train_2 \
--save_path checkpoint
"""

def build_net(input_shape: (int), output_size: int, dropout = 0.1):
	return	Sequential([
				Input(shape = input_shape),
				Dense(units = 32, activation="relu"),
				Dropout(dropout),
				Dense(units = 64, activation="relu"),
				Dropout(dropout),
				Dense(units = 32, activation="relu"),
				Dropout(dropout),
				Dense(units = output_size, activation="relu"),
			])

def build_router():
	return	IsolationForestRouter(
				init_params = {
					"n_estimators" 	: 200,
					"max_samples" 	: "auto"
				},
			)

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='SG ETA training of XG Boost models')
	parser.add_argument('--train_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--save_path', type = str, nargs = '?', help = 'Model checkpoint path')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# Define scaling rules
	# scaling_rules 	= 	[
	# 						OnlineMMDDrift(
	# 							min_trigger_count = 800,
	# 							safety_timestep = 128,
	# 							init_params = {
	# 								"ert" 			: 100,
	# 								"window_size" 	: 20,
	# 								"n_bootstraps" 	: 300,
	# 							}
	# 						)
	# 					]

	optimizer 		= 	tf.keras.optimizers.legacy.Adam(
							learning_rate = 1e-3,
						)

	loss_fn 		=  	tf.keras.losses.MeanSquaredError(
							reduction = tf.keras.losses.Reduction.SUM
						)

	model_wrapper 	= 	SupervisedModelWrapper(
							base_model 		= build_net(input_shape = 41, output_size = 1, dropout = 0),
							optimizer 		= optimizer,
							loss_fn 		= loss_fn,
							training_params = {
								"epochs" : 50,
							}, 
							training_batch_size = 64,
						)

	# Define checkpoint rules and policies
	checkpoint_rules = 	[ ]

	checkpoint_policy = DirectoryCheckpoint(save_path = args.save_path)

	model_router  	= 	build_router()

	compaction_rules = [ SizeRules(N = 5, K = 0) ]

	compaction_policy = EnsembleCompaction(N = 2, K = 3, strategy = "merge")

	scaling_rules   = [ BufferSizeLimit(min_size = 1) ]

	scaling_policy  = 	NaiveScaling(
							model 	= model_wrapper,
							router 	= model_router
						)

	expert_ensemble = ExpertEnsemble(
						buffer 				= LabelledFeatureBuffer(),
						scaling_rules 		= scaling_rules,
						scaling_policy 		= scaling_policy, 
						compaction_rules 	= compaction_rules,
						compaction_policy 	= compaction_policy,
						checkpoint_rules 	= checkpoint_rules,
						checkpoint_policy 	= checkpoint_policy
					)

	for file in glob.glob(f"{args.train_dir}/*.csv"):
		train_df = pd.read_csv(file)
		train_df = train_df.sample(frac = 1)

		feats_as_tensor   = tf.convert_to_tensor(train_df.drop("price", axis = 1).values, dtype = tf.float32)
		labels_as_tensor  = tf.convert_to_tensor(train_df["price"].values, dtype = tf.float32)
		labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])
		data_gen_training = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

		ingested_counts  = 0
		for batch_feats, batch_labels in data_gen_training.batch(64):
			batch_feats_wo_ts 	= batch_feats
			expert_ensemble.ingest(batch_input = (batch_feats_wo_ts, batch_labels))

			ingested_counts += len(batch_feats)
			logging.info(f"Total data ingested: {ingested_counts}, cur file: {file}, experts: {len(expert_ensemble.experts)}")

	if expert_ensemble.buffer.get_count() > 0:
		expert_ensemble.scale_experts()
		expert_ensemble.checkpoint_policy.save(expert_emsemble = expert_ensemble, log_state = True)