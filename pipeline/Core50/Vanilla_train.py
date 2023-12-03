import argparse
import logging
import glob
import os 
import tensorflow as tf
import time
import pickle
from adaptivestream.Models.Net import VggNet16Factory
from Examples.Math.index_tree_creation import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Conv2DTranspose, Reshape
from tqdm import tqdm

"""
python3 pipeline/Core50/Vanilla_train.py \
--train_dir /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/train \
--save_path checkpoint/core50/vgg/vanilla
"""

def build_net():
	return VggNet16Factory.get_model(input_shape = (128, 128, 3,), output_size = 10)

def save(model, save_path):
	current_time_round_up 	= int(time.time())
	model.save_weights(os.path.join(save_path, f"{current_time_round_up}_vanilla.h5"))
	return

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='Linear AdaptiveStream training on Core50')
	parser.add_argument('--train_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--save_path', type = str, nargs = '?', help = 'Model checkpoint path')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	optimizer 	= 	tf.keras.optimizers.legacy.Adam(
						learning_rate = 0.00005,
					)

	loss_fn 	= 	tf.keras.losses.SparseCategoricalCrossentropy(
						reduction = tf.keras.losses.Reduction.SUM
					)
	
	model 		= build_net()

	for each_file in sorted(glob.glob(f"{args.train_dir}/*.npy")):
		logging.info(f"Reading: {each_file}")

		train_dat 	= np.load(each_file, allow_pickle = True) # load
		np.random.shuffle(train_dat)
		train_dat   = train_dat

		for each_training_dat in tqdm(np.array_split(train_dat, 2)):
			feats_as_tensor   = tf.convert_to_tensor(each_training_dat[:,0].tolist(), dtype = tf.float32)
			labels_as_tensor  = tf.convert_to_tensor(each_training_dat[:,1].tolist(), dtype = tf.float32)

			dataset = tf.data.Dataset.from_tensor_slices((feats_as_tensor, labels_as_tensor)) \
									 .batch(32)

			model.compile(optimizer = optimizer, loss = loss_fn)
			model.fit(x = dataset, epochs = 30)
			
		save(model = model, save_path = args.save_path)