import argparse
import logging
import glob
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
--train_dir checkpoint/core50/vgg/linear/1701445892 \
--test_dir /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/test
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
			model = build_vgg_net()

		weight_path = os.path.join(path, f"{name}_model.h5")
		router_path = os.path.join(path, f"{name}_router.pkl")

		logging.info(f"Weights: {weight_path}, Router: {router_path}")

		model.load_weights(weight_path)
		with open(router_path, "r") as f:
			router = pickle.loads(f)

		expert 	= Expert(
					trained_model = model,
					router = router
				)

		if name == "fallback_model":
			dummy_expert_ensemble.fallback_expert = expert

		else:
			dummy_expert_ensemble.experts.append(expert)
	return dummy_expert_ensemble

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='Linear AdaptiveStream training on Core50')
	parser.add_argument('--net', type = str, nargs = '?', help = 'Network name')
	parser.add_argument('--train_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--test_dir', type = str, nargs = '?', help = 'Path to train features')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	expert_ensemble = load_model(path = args.train_dir, net = args.net)

	loss_fn 	= tf.keras.metrics.sparse_categorical_accuracy

	for each_file in sorted(glob.glob(f"{args.test_dir}/*.npy")):
		logging.info(f"Reading: {each_file}")
		train_dat 	= np.load(each_file, allow_pickle = True) # load

		feats_as_tensor   = tf.convert_to_tensor(each_training_dat[:,0].tolist(), dtype = tf.float32)
		labels_as_tensor  = tf.convert_to_tensor(each_training_dat[:,1].tolist(), dtype = tf.float32)
		labels_as_tensor  = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

		pred 		= expert_ensemble.infer(input_data = feats_as_tensor)
		acc 		= loss_fn(labels_as_tensor, pred)
		logging.info(f"File: {each_file}, Accuracy: {acc}")

	