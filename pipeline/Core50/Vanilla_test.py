import argparse
import logging
import glob
import os 
import tensorflow as tf
import time
import pickle
from adaptivestream.Models.Net import VggNet16Factory, CaffeNetFactory
from Examples.Math.index_tree_creation import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Conv2DTranspose, Reshape
from tqdm import tqdm

"""
python3 pipeline/Core50/Vanilla_test.py \
--net vgg \
--train_path checkpoint/core50/vgg/vanilla/1701584127_vanilla.h5 \
--test_dir /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/test
"""

def build_vgg_net():
	logging.info(f"Using VGGNet")
	return VggNet16Factory.get_model(input_shape = (128, 128, 3,), output_size = 10)

def build_caffe_net():
	logging.info(f"Using CaffeNet")
	return CaffeNetFactory.get_model(input_shape = (128, 128, 3,), output_size = 10)

def load_model(path: str, net: str):
	if net == "vgg":
		base_model 		= build_vgg_net()

	elif net == "caffe":
			base_model 	= build_caffe_net()

	base_model.load_weights(path)
	return base_model

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='Linear AdaptiveStream training on Core50')
	parser.add_argument('--net', type = str, nargs = '?', help = 'Network name')
	parser.add_argument('--train_path', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--test_dir', type = str, nargs = '?', help = 'Path to train features')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	model = load_model(path = args.train_path, net = args.net)

	loss_fn 	= tf.keras.metrics.sparse_categorical_accuracy
	batch_acc 	= []

	for each_file in sorted(glob.glob(f"{args.test_dir}/*.npy")):
		logging.info(f"Reading: {each_file}")
		train_dat 	= np.load(each_file, allow_pickle = True) # load

		feats_as_tensor   = tf.convert_to_tensor(train_dat[:,0].tolist(), dtype = tf.float32)
		labels_as_tensor  = tf.convert_to_tensor(train_dat[:,1].tolist(), dtype = tf.float32)

		pred = 	model(feats_as_tensor)

		correct_guesses = loss_fn(labels_as_tensor, pred)
		acc = sum(correct_guesses) / len(correct_guesses)
		batch_acc.append(acc)
		logging.info(f"File: {each_file}, Accuracy: {acc}")

	logging.info(f"Average accuracy: {np.mean(batch_acc)}")