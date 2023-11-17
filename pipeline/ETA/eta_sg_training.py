import argparse
import logging
import pandas as pd
import tensorflow as tf
from adaptivestream.Buffer import LabelledFeatureBuffer
from adaptivestream.Models import IndexedExpertEnsemble, IndexTreeBuilder, ExpertEnsemble
from adaptivestream.Models.Router import IsolationForestRouter, OutlierVAERouter, OneClassSVMRouter
from adaptivestream.Models.Wrapper import XGBoostModelWrapper
from adaptivestream.Policies.Compaction import NoCompaction
from adaptivestream.Policies.Scaling import NaiveScaling
from adaptivestream.Policies.Checkpoint import DirectoryCheckpoint
from adaptivestream.Rules.Scaling import BufferSizeLimit, TimeLimit, OnlineMMDDrift
from adaptivestream.Rules.Checkpoint import SaveOnStateChange
from alibi_detect.models.tensorflow.losses import elbo
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from Examples.Math.index_tree_creation import *

"""
python3 pipeline/ETA/eta_sg_training.py \
--train_path data/smpl_train_sg.csv \
--train_date_start '2023-01-14' \
--train_date_end '2023-01-21' \
--test_path data/smpl_train_sg.csv \
--save_path checkpoint
"""

def build_net():
	params = {	
				"n_estimators" : 100, 'min_child_weight': 1, 'learning_rate': 0.01, 'colsample_bytree': 0.3, 'max_depth': 10,
	            'subsample': 0.9, 'lambda': 0.7, 'nthread': -1, 'booster': 'gbtree', 
	            'gamma' : 0, 'eval_metric': 'mae', 'objective': 'reg:squarederror', 'seed' : 0, 'verbosity' : 1
            }
	return 	XGBRegressor(**params) 

def build_router():
	return	IsolationForestRouter(
				init_params = {
					"n_estimators" 	: 200,
					"max_samples" 	: "auto"
				},
			)

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='SG ETA training of XG Boost models')
	parser.add_argument('--train_path', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--train_date_start', type = str, nargs = '?', help = 'Start date for training')
	parser.add_argument('--train_date_end', type = str, nargs = '?', help = 'End date for training')
	parser.add_argument('--test_path', type = str, nargs = '?', help = 'Path to test features')
	parser.add_argument('--save_path', type = str, nargs = '?', help = 'Model checkpoint path')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# Define scaling rules
	# scaling_rules 	= 	[
	# 						TimeLimit(interval_sec = 7 * 24 * 60 * 60)
	# 					]

	scaling_rules 	= 	[
							OnlineMMDDrift(
								min_trigger_count = 800,
								safety_timestep = 256,
								init_params = {
									"ert" 			: 100,
									"window_size" 	: 20,
									"n_bootstraps" 	: 300,
								}
							)
						]
						
	model_wrapper 	= 	XGBoostModelWrapper(
							xg_boost_model 	= build_net(),
							training_params = {},
							loss_fn 		= tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM),
						)

	# Define checkpoint rules and policies
	checkpoint_rules = 	[
							SaveOnStateChange()
						]

	checkpoint_policy = DirectoryCheckpoint(save_path = args.save_path)

	model_router  	= 	build_router()

	scaling_policy  = 	NaiveScaling(
							model 	= model_wrapper,
							router 	= model_router
						)

	compaction_rules 	= []

	compaction_policy 	= NoCompaction()

	tree_builder 	= IndexTreeBuilder(
						leaf_expert_count = 3, 
						k_clusters = 2,
						exemplar_count = 3,
					)

	# expert_ensemble = IndexedExpertEnsemble(
	# 					tree_builder 		= tree_builder,
	# 					index_dim 			= 3,
	# 					buffer 				= LabelledFeatureBuffer(),
	# 					scaling_rules 		= scaling_rules,
	# 					scaling_policy 		= scaling_policy, 
	# 					compaction_rules 	= [],
	# 					compaction_policy 	= NoCompaction(),
	# 					checkpoint_rules 	= checkpoint_rules,
	# 					checkpoint_policy 	= checkpoint_policy
	# 				)	

	expert_ensemble = ExpertEnsemble(
						buffer 				= LabelledFeatureBuffer(),
						scaling_rules 		= scaling_rules,
						scaling_policy 		= scaling_policy, 
						compaction_rules 	= compaction_rules,
						compaction_policy 	= compaction_policy,
						checkpoint_rules 	= checkpoint_rules,
						checkpoint_policy 	= checkpoint_policy
					)

	train_df 	= pd.read_csv(args.train_path)
	train_df  	= train_df[
					(train_df.request_time >= args.train_date_start) & 
					(train_df.request_time <= args.train_date_end)
				]
	train_df["epoch_time"] 	= pd.to_datetime(train_df["request_time"], format = "%Y-%m-%d").map(pd.Timestamp.timestamp)
	train_df 				= train_df.drop("request_time", axis = 1)

	train_df.loc[train_df.index[-1], 'epoch_time'] = 16842253350

	feats_as_tensor   = tf.convert_to_tensor(train_df.drop("pred_diff", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(train_df["pred_diff"].values, dtype = tf.float32)
	labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])
	data_gen_training = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	ingested_counts  = 0
	for batch_feats, batch_labels in data_gen_training.batch(1):
		batch_feats_wo_ts 	= batch_feats[:, :-1]
		batch_ts  			= int(tf.math.reduce_max(batch_feats[:, -1]))

		expert_ensemble.ingest(	batch_input = (batch_feats_wo_ts, batch_labels), 
								batch_timestamp = batch_ts)
		
		ingested_counts += len(batch_feats)
		logging.info(f"Total data ingested: {ingested_counts}")