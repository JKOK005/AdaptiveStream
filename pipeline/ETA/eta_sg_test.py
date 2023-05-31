import argparse
import logging
import pandas as pd
import pickle
import tensorflow as tf

"""
python3 pipeline/ETA/eta_sg_test.py \
--model_path .pkl \
--test_path data/smpl_train_sg.csv \
--test_date '2023-01-28'
"""

if __name__ == "__main__":
	parser 	= argparse.ArgumentParser(description='SG ETA training of XG Boost models')
	parser.add_argument('--model_path', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--test_path', type = str, nargs = '?', help = 'Path to test features')
	parser.add_argument('--test_date', type = str, nargs = '?', help = 'Date filter for evaluating test')
	args 	= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	test_df  = pd.read_csv(args.test_path)
	test_df  = test_df[test_df.request_time == args.test_date]
	test_df  = test_df.drop("request_time", axis = 1)

	feats_as_tensor   = tf.convert_to_tensor(test_df.drop("pred_diff", axis = 1).values, dtype = tf.float32)
	labels_as_tensor  = tf.convert_to_tensor(test_df["pred_diff"].values, dtype = tf.float32)
	data_gen_testing  = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor))

	with open(args.model_path, "rb") as f:
		expert_ensemble = pickle.load(f)

	loss_fn  	= tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM) 
	batch_loss 	= 0
	batch_count = 0

	for batch_feats, batch_labels in data_gen_testing.batch(100000):
		row_count = batch_feats.shape[0]
		percentile_smpls = int(row_count * 0.05)
		
		feats_smpl 	= batch_feats[:percentile_smpls, :]
		labels_smpl = batch_labels[:percentile_smpls]

		pred = expert_ensemble.infer_w_smpls(input_data = batch_feats, 
											 truth_smpls = (feats_smpl, labels_smpl), 
											 alpha = 0.1)

		batch_loss 	+= loss_fn(batch_labels, pred)
		batch_count += 1

	logging.info(f"Average batch loss: {batch_loss / batch_count}")