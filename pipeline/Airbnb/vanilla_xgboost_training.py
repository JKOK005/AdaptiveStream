import argparse
import glob
import logging
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor

"""
python3 pipeline/Airbnb/vanilla_xgboost_training.py \
--train_dir data/airbnb/train_2 \
--test_dir data/airbnb/test_2
"""

def build_net():
	params = {	
				"n_estimators" : 1000, 'min_child_weight': 1, 'learning_rate': 0.01, 'colsample_bytree': 0.3, 'max_depth': 10,
	            'subsample': 0.9, 'lambda': 0.7, 'nthread': -1, 'booster': 'gbtree', 
	            'gamma' : 0, 'eval_metric': 'mae', 'objective': 'reg:squarederror', 'seed' : 0, 'verbosity' : 1
            }
	return 	XGBRegressor(**params) 

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='SG ETA training of XG Boost models')
	parser.add_argument('--train_dir', type = str, nargs='?', help='Path to train features')
	parser.add_argument('--test_dir', type = str, nargs='?', help='Path to test features')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
	logging.info(f"Using config: {args}")

	_dfs = []
	for file in glob.glob(f"{args.train_dir}/*.csv"):
		_dfs.append(pd.read_csv(file))

	train_df 	= pd.concat(_dfs)	
	print(f"Number training: {len(train_df)}")

	feats_as_tensor   = tf.convert_to_tensor(train_df.drop("price", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(train_df["price"].values, dtype = tf.float32)
	labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

	model 		= build_net()
	model.fit(feats_as_tensor, labels_as_tensor)

	_dfs = []
	for file in glob.glob(f"{args.test_dir}/*.csv"):
		_dfs.append(pd.read_csv(file))

	test_df 		  = pd.concat(_dfs)
	feats_as_tensor   = tf.convert_to_tensor(test_df.drop("price", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(test_df["price"].values, dtype = tf.float32)
	data_gen_testing  = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	loss_fn  	= tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) 
	batch_loss 	= 0
	batch_count = 0

	for batch_feats, batch_labels in data_gen_testing.batch(8):
		pred = model.predict(batch_feats)
		batch_loss 	+= loss_fn(batch_labels, pred)
		batch_count += 1

	logging.info(f"Test count: {len(test_df)}")
	logging.info(f"Average batch loss: {batch_loss / batch_count}")