import argparse
import logging
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor

"""
python3 pipeline/ETA/vanilla_xgboost_training.py \
--train_path data/smpl_train_sg.csv \
--test_path data/smpl_train_sg.csv \
--train_date_start '2023-01-14' \
--train_date_end '2023-01-21' \
--test_date_start '2023-01-27' \
--test_date_end '2023-01-28' \
--batch_size 64
"""

def build_net():
	params = {	
				"n_estimators" : 100, 'min_child_weight': 1, 'learning_rate': 0.01, 'colsample_bytree': 0.3, 'max_depth': 10,
	            'subsample': 0.9, 'lambda': 0.7, 'nthread': -1, 'booster': 'gbtree', 
	            'gamma' : 0, 'eval_metric': 'mae', 'objective': 'reg:squarederror', 'seed' : 0, 'verbosity' : 1
            }
	return 	XGBRegressor(**params) 

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='SG ETA training of XG Boost models')
	parser.add_argument('--train_path', type = str, nargs='?', help='Path to train features')
	parser.add_argument('--test_path', type = str, nargs='?', help='Path to test features')
	parser.add_argument('--train_date_start', type = str, nargs = '?', help = 'Start date for training')
	parser.add_argument('--train_date_end', type = str, nargs = '?', help = 'End date for training')
	parser.add_argument('--test_date_start', type = str, nargs = '?', help = 'Start date filter for evaluating test')
	parser.add_argument('--test_date_end', type = str, nargs = '?', help = 'End date filter for evaluating test')
	parser.add_argument('--batch_size', type = int, nargs = '?', help = 'Batch size for inference')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
	logging.info(f"Using config: {args}")

	train_df 	= pd.read_csv(args.train_path)
	train_df  	= train_df[
					(train_df.request_time >= args.train_date_start) & 
					(train_df.request_time <= args.train_date_end)
				]
	train_df 	= train_df.drop("request_time", axis = 1)

	print(f"Number training: {len(train_df)}")

	feats_as_tensor   = tf.convert_to_tensor(train_df.drop("pred_diff", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(train_df["pred_diff"].values, dtype = tf.float32)
	labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

	model 		= build_net()
	model.fit(feats_as_tensor, labels_as_tensor)

	df  		= pd.read_csv(args.test_path)
	df 			= df[(df.request_time >= args.test_date_start) & (df.request_time <= args.test_date_end)]

	for each_test_date in df.request_time.unique():
		test_df  		  = df[df.request_time == each_test_date]
		test_df 		  = test_df.drop("request_time", axis = 1)
		feats_as_tensor   = tf.convert_to_tensor(test_df.drop("pred_diff", axis = 1).values, dtype = tf.float32)
		labels_as_tensor  = tf.convert_to_tensor(test_df["pred_diff"].values, dtype = tf.float32)
		data_gen_testing  = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

		loss_fn  	= tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) 
		batch_loss 	= 0
		batch_count = 0

		for batch_feats, batch_labels in data_gen_testing.batch(args.batch_size):
			pred = model.predict(batch_feats)
			batch_loss 	+= loss_fn(batch_labels, pred)
			batch_count += 1

		logging.info(f"Test count: {len(test_df)}, date: {each_test_date}")
		logging.info(f"Average batch loss: {batch_loss / batch_count}")