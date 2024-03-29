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
python3 pipeline/Core50/Vanilla_lwf_train.py \
--net caffe \
--train_dir /workspace/jupyter_notebooks/adaptive-stream/data/Core50/save/NI/train \
--save_path checkpoint/core50/caffenet/vanilla_lwf/alpha_0.5
"""

def build_vgg_net():
	logging.info(f"Using VGGNet")
	return VggNet16Factory.get_model(input_shape = (128, 128, 3,), output_size = 10)

def build_caffe_net():
	logging.info(f"Using CaffeNet")
	return CaffeNetFactory.get_model(input_shape = (128, 128, 3,), output_size = 10)

def save(model, save_path):
	current_time_round_up 	= int(time.time())
	model.save_weights(os.path.join(save_path, f"{current_time_round_up}_vanilla.h5"))
	return

class LwfLoss(tf.keras.losses.Loss):
	cur_loss 	= tf.keras.losses.SparseCategoricalCrossentropy(
					reduction = tf.keras.losses.Reduction.SUM
				)

	prior_loss 	= tf.keras.losses.KLDivergence(
					reduction = tf.keras.losses.Reduction.SUM
				)


	def __init__(self, 	tmp: float, lwf_alpha: float, 
						*args, **kwargs):
		super(LwfLoss, self).__init__(*args, **kwargs)
		self.prior_y_pred 	= None 
		self.tmp 			= tmp
		self.lwf_alpha 		= lwf_alpha
		return

	def call(self, y_true, y_pred):
		if self.prior_y_pred is None:
			self.prior_y_pred = y_pred

		# cur_loss 	= tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
		# prior_loss 	= tf.keras.losses.kullback_leibler_divergence(self.prior_y_pred, y_pred)

		_cur_loss 	= self.cur_loss(y_true, y_pred)
		_prior_loss = self.prior_loss(self.prior_y_pred, y_pred)

		lwf_loss = 	(1 - self.lwf_alpha) * _cur_loss + \
					(self.lwf_alpha * (self.tmp ** 2)) * _prior_loss

		self.prior_y_pred = tf.get_static_value(y_pred)
		return lwf_loss

if __name__ == "__main__":
	parser 		= argparse.ArgumentParser(description='Linear AdaptiveStream training on Core50')
	parser.add_argument('--net', type = str, nargs = '?', help = 'Type of network')
	parser.add_argument('--train_dir', type = str, nargs = '?', help = 'Path to train features')
	parser.add_argument('--save_path', type = str, nargs = '?', help = 'Model checkpoint path')
	args 		= parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	optimizer 	= 	tf.keras.optimizers.legacy.Adam(
						learning_rate = 0.00005,
					)

	loss_fn 	= LwfLoss(tmp = 1.1, lwf_alpha = 0.5)

	model 		= build_vgg_net() if args.net == "vgg" else build_caffe_net()

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