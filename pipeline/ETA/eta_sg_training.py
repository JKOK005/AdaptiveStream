import argparse
import logging
import pandas as pd
import tensorflow as tf
from adaptivestream.Buffer import LabelledFeatureBuffer
from adaptivestream.Models import IndexedExpertEnsemble, IndexTreeBuilder
from adaptivestream.Models.Router import OutlierVAERouter, OneClassSVMRouter
from adaptivestream.Models.Wrapper import XGBoostModelWrapper
from adaptivestream.Policies.Compaction import NoCompaction
from adaptivestream.Policies.Scaling import NaiveScaling
from adaptivestream.Rules.Scaling import BufferSizeLimit
from alibi_detect.models.tensorflow.losses import elbo
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

"""
python3 pipeline/ETA/eta_sg_training.py \
--train_path data/smpl_train_sg.csv \
--test_path data/smpl_train_sg.csv \
--buffer_limit 1800
"""

def build_net():
	params = {	
				"n_estimators" : 10000, 'min_child_weight': 1, 'learning_rate': 0.001, 'colsample_bytree': 0.3, 'max_depth': 10,
	            'subsample': 0.9, 'lambda': 0.7, 'nthread': -1, 'booster': 'gbtree', 
	            'gamma' : 0, 'eval_metric': 'mae', 'objective': 'reg:squarederror', 'seed' : 0, 'verbosity' : 1
            }
	return 	XGBRegressor(**params) 

def build_router():
	return	OneClassSVMRouter(
				init_params = {
					"kernel" 	: "rbf",
					"degree" 	: 6,
					"max_iter" 	: 5000
				},
			)

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='SG ETA training of XG Boost models')
	parser.add_argument('--train_path', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--test_path', type = str, nargs = '?', help = 'Path to test features')
	parser.add_argument('--buffer_limit', type = int, nargs = '?', help = 'Buffer limit size')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# Define scaling rules
	scaling_rules 	= 	[
							BufferSizeLimit(min_size = args.buffer_limit)
						]	

	# Define scaling policy
	model_wrapper 	= 	XGBoostModelWrapper(
							xg_boost_model 	= build_net(),
							training_params = {},
						)

	# model_router  	= 	build_router(input_size = 8)
	model_router  	= 	build_router()

	scaling_policy  = 	NaiveScaling(
							model 	= model_wrapper,
							router 	= model_router
						)

	tree_builder 	= IndexTreeBuilder(
						leaf_expert_count = 3, 
						k_clusters = 2,
						exemplar_count = 3,
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

	train_df 	= pd.read_csv(args.train_path)
	train_df 	= train_df.drop("request_time", axis = 1)

	feats_as_tensor   = tf.convert_to_tensor(train_df.drop("pred_diff", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(train_df["pred_diff"].values, dtype = tf.float32)
	labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])
	data_gen_training = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	test_df 	= pd.read_csv(args.test_path)
	test_df 	= test_df.drop("request_time", axis = 1)

	feats_as_tensor   = tf.convert_to_tensor(test_df.drop("pred_diff", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(test_df["pred_diff"].values, dtype = tf.float32)
	data_gen_testing  = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	ingested_counts  = 0
	for batch_feats, batch_labels in data_gen_training.batch(100):
		expert_ensemble.ingest(batch_input = (batch_feats, batch_labels))
		ingested_counts += len(batch_feats)
		logging.info(f"Total data ingested: {ingested_counts}")

	loss_fn  	= tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM) 
	batch_loss 	= 0
	batch_count = 0

	for batch_feats, batch_labels in data_gen_testing.batch(100):
		pred = expert_ensemble.infer(input_data = batch_feats)
		batch_loss 	+= loss_fn(batch_labels, pred)
		batch_count += 1

	logging.info(f"Average batch loss: {batch_loss / batch_count}")