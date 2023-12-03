import argparse
import logging
import glob
import numpy as np
import os 
import tensorflow as tf
import pickle
from pathlib import Path
from adaptivestream.Models import ExpertEnsemble, Expert
from adaptivestream.Models.Router import OutlierAERouter
from adaptivestream.Models.Wrapper import SupervisedModelWrapper
from adaptivestream.Models.Net import VggNet16Factory
from adaptivestream.Rules.Scaling import OnlineMMDDrift
from adaptivestream.Policies.Scaling import NaiveScaling
from adaptivestream.Policies.Compaction import NoCompaction
from tqdm import tqdm

"""
python3 pipeline/Core50/Linear_adaptive_test.py \
--net vgg \
--train_dir checkpoint/core50/vgg/linear/1701443998 \
--test_dir /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/test \
--alpha 0.1
"""

def build_vgg_net():
	return VggNet16Factory.get_model(input_shape = (128, 128, 3,), output_size = 10)

def load_model(path: str, net: str):
	dummy_expert_ensemble = ExpertEnsemble(
								buffer = None,
								scaling_rules = None,
								scaling_policy = NaiveScaling(model = None, router = None),
								compaction_rules = None, 
								compaction_policy = NoCompaction(),
							)

	for each_weight_file in sorted(glob.glob(f"{path}/*.h5")):
		name = each_weight_file.split("/")[-1].replace("_model.h5", "")

		if net == "vgg":
			base_model = build_vgg_net()

		weight_path = os.path.join(path, f"{name}_model.h5")
		router_path = os.path.join(path, f"{name}_router.pkl")

		logging.info(f"Weights: {weight_path}, Router: {router_path}")

		base_model.load_weights(weight_path)
		with open(router_path, "rb") as f:
			router = pickle.load(f)

		model_wrapper 	= 	SupervisedModelWrapper(
								base_model 		= base_model,
								optimizer 		= None,
								loss_fn 		= None,
								training_params = None, 
								training_batch_size = None,
							)

		expert 	= Expert(
					trained_model = model_wrapper,
					router = router
				)

		if name == "fallback":
			dummy_expert_ensemble.fallback_expert = expert

		else:
			dummy_expert_ensemble.experts.append(expert)
	return dummy_expert_ensemble

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='Linear AdaptiveStream training on Core50')
	parser.add_argument('--net', type = str, nargs = '?', help = 'Network name')
	parser.add_argument('--train_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--test_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--alpha', type = float, nargs = '?', help = 'Weighted ratio of using outlier loss to sample loss')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	expert_ensemble = load_model(path = args.train_dir, net = args.net)

	loss_fn 	= tf.keras.metrics.sparse_categorical_accuracy
	batch_acc 	= []

	for each_file in sorted(glob.glob(f"{args.test_dir}/*.npy")):
		logging.info(f"Reading: {each_file}")
		train_dat 	= np.load(each_file, allow_pickle = True) # load

		feats_as_tensor   = tf.convert_to_tensor(train_dat[:,0].tolist(), dtype = tf.float32)
		labels_as_tensor  = tf.convert_to_tensor(train_dat[:,1].tolist(), dtype = tf.float32)

		row_count = feats_as_tensor.shape[0]
		row_smpls = int(row_count * 0.1)

		feats_smpl 	= feats_as_tensor[:row_smpls, :]
		labels_smpl = labels_as_tensor[:row_smpls]

		pred = expert_ensemble.infer_w_smpls(
									input_data = feats_as_tensor, 
									truth_smpls = (feats_smpl, labels_smpl), 
									alpha = args.alpha
								)

		correct_guesses = loss_fn(labels_as_tensor, pred)
		acc = sum(correct_guesses) / len(correct_guesses)
		batch_acc.append(acc)

		logging.info(f"File: {each_file}, Accuracy: {acc}")
	logging.info(f"Average accuracy: {np.mean(batch_acc)}")